import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class FrozenLake8x8Env(gym.Env):
    metadata = {'render_modes': ['pygame'], 'render_fps': 5}
    
    def __init__(self):
        super().__init__()
        self.grid_size = 8
        
        #Grid: S=Start, N=Nothing, M=Meteorite, G=Goal, P=Palm
        self.grid = np.array([
            ['S','N','N','M','N','N','P','N'],
            ['N','M','N','N','M','N','N','N'],
            ['N','N','M','N','N','M','N','P'],
            ['M','N','N','M','N','N','M','N'],
            ['N','M','N','P','M','N','N','N'],
            ['N','N','M','N','N','M','N','N'],
            ['N','M','N','N','M','N','N','N'],
            ['N','N','M','N','N','P','M','G']
        ])
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.agent_pos = (0,0)
        
        self.screen = None
        self.cell_size = 100
        self.clock = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = (0,0)
        return self._get_state(), {}
    
    def _get_state(self):
        i,j = self.agent_pos
        return i*self.grid_size + j
    
    def step(self, action):
        i,j = self.agent_pos
        if action == 0 and i>0: i -= 1
        if action == 1 and j<self.grid_size-1: j += 1
        if action == 2 and i<self.grid_size-1: i += 1
        if action == 3 and j>0: j -= 1
        self.agent_pos = (i,j)
        
        cell = self.grid[i,j]
        reward, done = 0, False
        if cell == 'M': reward, done = -10, True
        elif cell == 'G': reward, done = 10, True
        elif cell == 'P': reward = 5
        
        return self._get_state(), reward, done, False, {}
    
    def render(self, mode='pygame'):
        if mode != 'pygame':
            raise NotImplementedError("Solo modo pygame implementado")
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size*self.cell_size, self.grid_size*self.cell_size))
            pygame.display.set_caption("FrozenLake 8x8")
            self.clock = pygame.time.Clock()
        
        COLORS = {
            'N': (0,255,0),  # light blue
        }

        self.images = {
            'S': pygame.image.load("imagenes/grass.png"),
            'M': pygame.image.load("imagenes/meteorito.png"),
            'G': pygame.image.load("imagenes/cueva.png"),
            'P': pygame.image.load("imagenes/palmera.png")
        }
        self.agent_img = pygame.image.load("imagenes/dino.png")
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, COLORS[self.grid[i,j]], rect)
                pygame.draw.rect(self.screen, (255,255,255), rect, 1)
        
        ai,aj = self.agent_pos
        agent_rect = pygame.Rect(aj*self.cell_size, ai*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, AGENT_COLOR, agent_rect)
        
        pygame.display.flip()
        self.clock.tick(5)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


env = FrozenLake8x8Env()
state,_ = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # mover aleatoriamente
    state, reward, done, _, _ = env.step(action)
    env.render()

env.close()
