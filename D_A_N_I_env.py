import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class DaniEnv(gym.Env):
    metadata = {'render_modes': ['pygame'], 'render_fps': 5}
    
    def __init__(self, grid_list=None):
        super().__init__()
        self.grid_size = 8
        self.cell_size = 100
        
        # Si no se proporciona lista de grids, usar el original más algunos adicionales
        if grid_list is None:
            self.grid_list = self._create_additional_grids()
        else:
            self.grid_list = grid_list
            
        self.current_grid_idx = 0
        self.original_grid = self.grid_list[self.current_grid_idx].copy()
        self.grid = self.original_grid.copy()
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.agent_pos = (0,0)
        
        # Pygame variables
        self.screen = None
        self.clock = None
        
        # Cargar imágenes y escalarlas
        self.images = {
            'S': pygame.transform.scale(pygame.image.load("env_images/grass.png"), (self.cell_size, self.cell_size)),
            'M': pygame.transform.scale(pygame.image.load("env_images/meteorito.png"), (self.cell_size, self.cell_size)),
            'G': pygame.transform.scale(pygame.image.load("env_images/cueva.png"), (self.cell_size, self.cell_size)),
            'P': pygame.transform.scale(pygame.image.load("env_images/palmera.png"), (self.cell_size, self.cell_size))
        }
        self.agent_img = pygame.transform.scale(pygame.image.load("env_images/dino.png"), (self.cell_size, self.cell_size))
    
    def _create_additional_grids(self):
        """Crear grids adicionales manteniendo el original como primero"""
        grids = []
        
        #Grid 0 (Muy fácil) - Camino directo con muchas palmeras
        grids.append(np.array([
            ['S','N','N','N','N','N','P','N'],
            ['P','N','N','N','M','N','N','N'],
            ['P','P','M','N','N','N','N','P'],
            ['N','P','N','N','N','N','M','N'],
            ['N','P','P','P','P','P','N','N'],
            ['N','N','N','N','N','P','N','N'],
            ['N','M','N','N','N','P','N','N'],
            ['N','N','N','N','N','P','P','G']
        ]))

        #Grid 1 (Fácil) - Camino sin muchos meteoritos y pocas palmeras
        grids.append(np.array([
            ['S','N','N','N','N','N','P','N'],
            ['N','N','N','N','M','N','N','N'],
            ['N','N','M','N','N','N','N','P'],
            ['M','N','N','N','N','N','M','N'],
            ['N','M','N','P','N','N','N','N'],
            ['N','N','N','N','N','M','N','N'],
            ['N','M','N','N','N','N','N','N'],
            ['N','N','M','N','N','P','N','G']
        ]))
        
        
        #Grid 2 (Fácil) - Camino sin muchos meteoritos y pocas palmeras
        grids.append(np.array([
            ['S','N','N','N','N','N','P','N'],
            ['N','N','N','N','M','N','N','N'],
            ['N','N','M','N','N','N','N','P'],
            ['N','N','N','N','N','N','M','N'],
            ['N','M','N','P','N','N','N','N'],
            ['N','N','N','N','N','M','N','N'],
            ['N','M','N','N','N','N','N','N'],
            ['N','N','N','N','N','P','M','G']
        ]))


        #Grid 3 (Difícil) - Muchos meteoritos
        grids.append(np.array([
            ['S','N','N','M','N','N','P','N'],
            ['N','M','N','N','M','N','N','N'],
            ['N','N','M','N','N','M','N','P'],
            ['M','N','N','M','N','N','M','N'],
            ['N','M','N','P','M','N','N','N'],
            ['N','N','M','N','N','M','N','N'],
            ['N','M','N','N','M','N','N','N'],
            ['N','N','M','N','N','P','M','G']
        ]))

        #Grid 4 (Difícil) - Muchos meteoritos y palmeras
        grids.append(np.array([
            ['S','N','P','M','N','N','P','N'],
            ['N','M','N','N','M','N','N','N'],
            ['P','N','M','N','P','M','N','P'],
            ['M','N','N','M','N','N','M','N'],
            ['N','M','N','P','M','N','N','N'],
            ['N','P','M','N','N','M','N','N'],
            ['N','M','N','N','M','N','P','N'],
            ['N','N','M','N','N','P','M','G']
        ]))
        
        
        
        return grids
        
    def set_grid(self, grid_idx):
        """Establecer un grid específico"""
        self.current_grid_idx = grid_idx
        self.original_grid = self.grid_list[self.current_grid_idx].copy()
        self.grid = self.original_grid.copy()
        
        # Encontrar la posición inicial (S) en el grid actual
        start_positions = np.where(self.grid == 'S')
        if len(start_positions[0]) > 0:
            self.agent_pos = (start_positions[0][0], start_positions[1][0])
        else:
            self.agent_pos = (0, 0)  # Fallback si no hay S
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Restaurar el grid actual (no cambiar de grid)
        self.grid = self.original_grid.copy()
        
        # Encontrar la posición inicial (S) en el grid actual
        start_positions = np.where(self.grid == 'S')
        if len(start_positions[0]) > 0:
            self.agent_pos = (start_positions[0][0], start_positions[1][0])
        else:
            self.agent_pos = (0, 0)  # Fallback si no hay S
        
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
            reward, done = 100, True
        elif cell == 'P': 
            reward = 20  # bonificación por palmera
            self.grid[i,j] = 'N'  # la palmera se come y desaparece
        else:  # celda N o S
            reward = -1  # castigo pequeño por moverse

        return self._get_state(), reward, done, False, {}
    
    def render(self, mode='pygame'):
        if mode != 'pygame':
            raise NotImplementedError("Solo modo pygame implementado")
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size*self.cell_size, self.grid_size*self.cell_size))
            pygame.display.set_caption(f"FrozenLake 8x8 - Grid {self.current_grid_idx+1}")
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

