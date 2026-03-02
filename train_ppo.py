"""
Territorial.io PPO Training - Single File for Kaggle
====================================================
Run this on Kaggle with GPU enabled.

Author: AI Assistant
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    episodes = 5000
    max_steps_per_episode = 2000
    batch_size = 64
    ppo_epochs = 10
    ppo_clip_eps = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    hidden_size = 256
    learning_rate = 3e-4
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    
    map_size = 20
    initial_balance = 100
    opening_phase_seconds = 107
    max_balance_cap = 5000
    defender_advantage = 2.0
    
    reward_win = 100.0
    reward_loss = -50.0
    reward_territory_gain = 1.0
    
    log_interval = 10
    model_save_path = "/kaggle/working/models"


# ============================================================================
# GAME ENVIRONMENT
# ============================================================================

class Tile:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.owner_id = None
        self.balance = 0
        self.is_ocean = False


class Player:
    def __init__(self, player_id, is_bot=True):
        self.id = player_id
        self.is_bot = is_bot
        self.balance = Config.initial_balance
        self.territory = []
        self.alive = True
        self.is_eliminated = False


class GameEnv:
    def __init__(self, num_bots=19):
        self.num_bots = num_bots
        self.total_players = num_bots + 1
        self.reset()
    
    def reset(self):
        self.map = [[Tile(x, y) for y in range(Config.map_size)] 
                    for x in range(Config.map_size)]
        
        # Add oceans
        for x in range(Config.map_size):
            self.map[x][0].is_ocean = True
            self.map[x][Config.map_size-1].is_ocean = True
        for y in range(Config.map_size):
            self.map[0][y].is_ocean = True
            self.map[Config.map_size-1][y].is_ocean = True
        
        self.players = []
        self.ai_player = Player(0, is_bot=False)
        self.players.append(self.ai_player)
        
        for i in range(1, self.total_players):
            self.players.append(Player(i, is_bot=True))
        
        # Spawn
        available = [t for row in self.map for t in row if not t.is_ocean]
        random.shuffle(available)
        
        for p in self.players:
            if available:
                t = available.pop(0)
                t.owner_id = p.id
                p.territory.append(t)
        
        self.time_elapsed = 0.0
        self.game_over = False
        self.winner = None
        self.max_territory = 0
        
        return self._get_state()
    
    def step(self, action):
        target_id, pct = action
        
        # AI action
        if target_id == -1:
            self._expand(self.ai_player, pct)
        else:
            self._attack(self.ai_player, target_id, pct)
        
        # Bot actions
        for bot in self.players[1:]:
            if bot.alive:
                tgt, p = self._bot_decision(bot)
                if tgt is None:
                    self._expand(bot, p)
                else:
                    self._attack(bot, tgt, p)
        
        # Interest
        self.time_elapsed += 1
        if int(self.time_elapsed) % 5 == 0:
            self._apply_interest()
        
        # Check elimination
        for p in self.players:
            if len(p.territory) == 0 and p.balance < 10:
                p.is_eliminated = True
                p.alive = False
        
        # Check game over
        alive = [p for p in self.players if p.alive and not p.is_eliminated]
        if len(alive) == 1:
            self.game_over = True
            self.winner = alive[0].id
        
        if self.ai_player.is_eliminated:
            self.game_over = True
        
        reward = self._get_reward()
        
        return self._get_state(), reward, self.game_over, {}
    
    def _expand(self, player, pct):
        if not player.alive:
            return
        
        troops = int(player.balance * pct / 100)
        if troops < 1:
            return
        
        player.balance -= troops
        
        # Find empty neighbors
        empty = []
        for t in player.territory:
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = t.x + dx, t.y + dy
                if 0 <= nx < Config.map_size and 0 <= ny < Config.map_size:
                    adj = self.map[nx][ny]
                    if adj.owner_id is None and not adj.is_ocean:
                        empty.append(adj)
        
        if empty:
            for _ in range(min(3, len(empty))):
                if empty:
                    t = empty.pop(0)
                    t.owner_id = player.id
                    player.territory.append(t)
    
    def _attack(self, attacker, target_id, pct):
        if not attacker.alive:
            return
        
        target = self.players[target_id]
        if not target.alive:
            return
        
        troops = int(attacker.balance * pct / 100)
        if troops < 1:
            return
        
        attacker.balance -= troops
        
        # Battle (defender advantage 2:1)
        defender_loss = troops // 2
        
        if len(target.territory) > 0:
            for _ in range(min(defender_loss, len(target.territory))):
                if target.territory:
                    t = random.choice(target.territory)
                    t.owner_id = attacker.id
                    attacker.territory.append(t)
                    target.territory.remove(t)
    
    def _bot_decision(self, bot):
        enemies = []
        for t in bot.territory:
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = t.x + dx, t.y + dy
                if 0 <= nx < Config.map_size and 0 <= ny < Config.map_size:
                    adj = self.map[nx][ny]
                    if adj.owner_id is not None and adj.owner_id != bot.id:
                        e = self.players[adj.owner_id]
                        if e.alive and not e.is_eliminated:
                            enemies.append(e.id)
        
        if not enemies:
            return None, 25
        
        target = random.choice(enemies)
        
        if self.time_elapsed < 107:
            return target, 20
        else:
            return target, 30
    
    def _apply_interest(self):
        rate = max(0, 0.5 * (1 - self.time_elapsed / 120))
        for p in self.players:
            if p.alive and not p.is_eliminated:
                bonus = len(p.territory) * 10
                max_cap = min(Config.max_balance_cap, 100 + bonus)
                p.balance += int(min(p.balance, max_cap) * rate)
    
    def _get_reward = 0
        curr = len(self):
        r(self.ai_player.territory)
        r += (curr - self.max_territory) * Config.reward_territory_gain
        self.max_territory = max(self.max_territory, curr)
        r += self.ai_player.balance * 0.001
        r += 0.01
        
        if self.game_over:
            if self.winner == 0:
                r += Config.reward_win
            else:
                r += Config.reward_loss
        
        return r
    
    def _get_state(self):
        s = []
        s.append(min(self.ai_player.balance / Config.max_balance_cap, 1.0))
        s.append(len(self.ai_player.territory) / 400)
        s.append(self.time_elapsed / 300)
        
        alive = sum(1 for p in self.players if p.alive and not p.is_eliminated)
        s.append(alive / self.total_players)
        
        while len(s) < 50:
            s.append(0.0)
        
        return np.array(s[:50], dtype=np.float32)
    
    def get_valid_actions(self):
        actions = [(-1, p) for p in range(10, 100, 10)]
        
        for bot in self.players[1:]:
            if bot.alive and not bot.is_eliminated:
                for p in range(10, 100, 10):
                    actions.append((bot.id, p))
        
        return actions


