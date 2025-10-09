import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import time

class FrozenLake8x8Env(gym.Env):
    metadata = {'render_modes': ['pygame'], 'render_fps': 5}
    
    def __init__(self):
        super().__init__()
        self.grid_size = 8
        self.cell_size = 100
        
        # Grid original: S=Start, N=Nothing, M=Meteorite, G=Goal, P=Palm
        self.original_grid = np.array([
            ['S','N','N','M','N','N','P','N'],
            ['N','M','N','N','M','N','N','N'],
            ['N','N','M','N','N','M','N','P'],
            ['M','N','N','M','N','N','M','N'],
            ['N','M','N','P','M','N','N','N'],
            ['N','N','M','N','N','M','N','N'],
            ['N','M','N','N','M','N','N','N'],
            ['N','N','M','N','N','P','M','G']
        ])
        
        # Grid que se modifica durante el episodio
        self.grid = self.original_grid.copy()
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.agent_pos = (0,0)
        
        # Pygame variables
        self.screen = None
        self.clock = None
        
        # Cargar imágenes y escalarlas
        self.images = {
            'S': pygame.transform.scale(pygame.image.load("imagenes/grass.png"), (self.cell_size, self.cell_size)),
            'M': pygame.transform.scale(pygame.image.load("imagenes/meteorito.png"), (self.cell_size, self.cell_size)),
            'G': pygame.transform.scale(pygame.image.load("imagenes/cueva.png"), (self.cell_size, self.cell_size)),
            'P': pygame.transform.scale(pygame.image.load("imagenes/palmera.png"), (self.cell_size, self.cell_size))
        }
        self.agent_img = pygame.transform.scale(pygame.image.load("imagenes/dino.png"), (self.cell_size, self.cell_size))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = (0,0)
        # Restaurar el grid original para cada episodio
        self.grid = self.original_grid.copy()
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
        if cell == 'M': 
            reward, done = -100, True
        elif cell == 'G': 
            reward, done = 20, True
        elif cell == 'P': 
            reward = 10  # bonificación por palmera
            self.grid[i,j] = 'N'  # la palmera se come y desaparece
        else:  # celda N o S
            reward = -0.5  # castigo pequeño por moverse

        return self._get_state(), reward, done, False, {}
    
    def render(self, mode='pygame'):
        if mode != 'pygame':
            raise NotImplementedError("Solo modo pygame implementado")
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size*self.cell_size, self.grid_size*self.cell_size))
            pygame.display.set_caption("FrozenLake 8x8")
            self.clock = pygame.time.Clock()
        
        # Dibujar grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size)
                cell_type = self.grid[i,j]
                
                # Color de fondo
                if cell_type == 'N':
                    pygame.draw.rect(self.screen, (114, 184, 68), rect)  # verde claro
                elif cell_type == "M":
                    pygame.draw.rect(self.screen, (94, 170, 246), rect)  # azul
                elif cell_type == "P":
                    pygame.draw.rect(self.screen, (57, 119, 22), rect)  # verde oscuro
                else:
                    pygame.draw.rect(self.screen, (0,0,0), rect)  # gris neutro para S,G

                # Dibujar imagen de la celda
                if cell_type in self.images:
                    self.screen.blit(self.images[cell_type], rect)
                
                pygame.draw.rect(self.screen, (255,255,255), rect, 1)  # borde blanco
        
        # Dibujar agente
        ai,aj = self.agent_pos
        agent_rect = pygame.Rect(aj*self.cell_size, ai*self.cell_size, self.cell_size, self.cell_size)
        self.screen.blit(self.agent_img, agent_rect)
        
        pygame.display.flip()
        self.clock.tick(5)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ---------------------
# Entrenamiento con Q-learning
# ---------------------
env = FrozenLake8x8Env()

# Parámetros Q-learning
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01  # Reducir el mínimo para más exploración
epsilon_decay = 0.999  # Decay más lento
episodes = 10000  # Más episodios

# Inicializar tabla Q
Q = np.zeros((env.observation_space.n, env.action_space.n))
rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 100:  # Límite de pasos por episodio
        # Epsilon-greedy
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done, _, _ = env.step(action)
        
        # Actualizar Q con mejor fórmula
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next_action]
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error
        
        state = next_state
        total_reward += reward
        steps += 1

    # Exploración adicional ocasional
    if ep % 100 == 0:
        epsilon = max(epsilon_min, epsilon * 0.9)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    rewards.append(total_reward)
    
    
print("Entrenamiento terminado.")
print("Promedio de recompensas últimos 100 episodios:", np.mean(rewards[-100:]))

# ------------------------------
# Mostrar el agente jugando
# ------------------------------
obs, _ = env.reset()
done = False
while not done:
    action = np.argmax(Q[obs])  # política greedy
    obs, reward, done, _, _ = env.step(action)
    env.render()
    time.sleep(0.3)  # animación lenta para verlo

env.close()
