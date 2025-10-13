import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import time

class FrozenLake8x8Env(gym.Env):
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
            'S': pygame.transform.scale(pygame.image.load("imagenes/grass.png"), (self.cell_size, self.cell_size)),
            'M': pygame.transform.scale(pygame.image.load("imagenes/meteorito.png"), (self.cell_size, self.cell_size)),
            'G': pygame.transform.scale(pygame.image.load("imagenes/cueva.png"), (self.cell_size, self.cell_size)),
            'P': pygame.transform.scale(pygame.image.load("imagenes/palmera.png"), (self.cell_size, self.cell_size))
        }
        self.agent_img = pygame.transform.scale(pygame.image.load("imagenes/dino.png"), (self.cell_size, self.cell_size))
    
    def _create_additional_grids(self):
        """Crear grids adicionales manteniendo el original como primero"""
        grids = []
        
        # Grid original (el mismo que tenías)
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
        
        # Grid 2 - Variación con menos meteoritos
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
        
        # Grid 3 - Variación con más palmeras
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
        
        # Grid 4 - Variación con camino más directo
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
        # Grid 4 - Variación con camino más directo
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


# ---------------------
# Entrenamiento SECUENCIAL con reinicio de Q-table
# ---------------------
env = FrozenLake8x8Env()

# Parámetros Q-learning
alpha = 0.2
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.998
episodes_per_grid = 6000  # Episodios por grid

# Almacenar resultados por grid
grid_results = []

print(f"Entrenando secuencialmente en {len(env.grid_list)} grids diferentes...")

# Para cada grid, entrenar desde cero
for grid_idx in range(len(env.grid_list)):
    print(f"\n=== COMENZANDO ENTRENAMIENTO EN GRID {grid_idx+1} ===")
    
    # Establecer el grid actual
    env.set_grid(grid_idx)
    
    # REINICIAR la tabla Q para este grid (empezar desde cero)
    Q = np.random.uniform(-0.1, 0.1, (env.observation_space.n, env.action_space.n))
    
    # Reiniciar parámetros de exploración
    epsilon = 1.0
    rewards = []
    success_count = 0
    
    # Entrenar en este grid específico
    for ep in range(episodes_per_grid):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 100:
            # Epsilon-greedy
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _, _ = env.step(action)
            
            # Actualizar Q
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Contar éxitos
            if done and reward > 0:
                success_count += 1
        
        # Decaimiento de epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)
        
        # Mostrar progreso cada 500 episodios
        if ep % 500 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            success_rate = success_count / 500 if ep > 0 else 0
            print(f"Grid {grid_idx+1}, Episodio {ep}: Recompensa promedio = {avg_reward:.2f}, Éxitos = {success_count}, Epsilon = {epsilon:.3f}")
            success_count = 0
    
    # Guardar resultados de este grid
    final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    grid_results.append({
        'grid_idx': grid_idx,
        'final_avg_reward': final_avg,
        'Q_table': Q.copy()  # Guardar la Q-table aprendida para este grid
    })
    
    print(f"\n--- ENTRENAMIENTO COMPLETADO PARA GRID {grid_idx+1} ---")
    print(f"Recompensa promedio últimos 100 episodios: {final_avg:.2f}")
    
    # Demostrar el agente en este grid inmediatamente después del entrenamiento
    print(f"\nDemostración en Grid {grid_idx+1}:")
    env.set_grid(grid_idx)  # Asegurarse de que estamos en el grid correcto
    obs = env._get_state()
    done = False
    steps = 0
    
    while not done and steps < 50:
        action = np.argmax(Q[obs])  # política greedy
        obs, reward, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.3)
        steps += 1
        
        if done:
            if reward > 0:
                print(f"¡Éxito! Recompensa: {reward}")
            else:
                print(f"Fracaso. Recompensa: {reward}")
            break
    
    if not done:
        print(f"Tiempo agotado después de {steps} pasos")

# Resumen final
print("\n" + "="*50)
print("RESUMEN FINAL DEL ENTRENAMIENTO")
print("="*50)
for result in grid_results:
    print(f"Grid {result['grid_idx']+1}: Recompensa promedio = {result['final_avg_reward']:.2f}")

env.close()