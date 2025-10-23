import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class DaniEnv(gym.Env):
    """  
    The agent moves through an 8x8 grid trying to reach the goal while avoiding obstacles.
    Different cell types have different effects:
    - 'S' : Start position
    - 'N': Normal cell (safe to move)
    - 'M': Meteorite (ends episode with negative reward)
    - 'G': Goal/Cave (ends episode with positive reward)
    - 'P': Palm tree (gives positive reward and disappears when collected)
    
    The environment includes multiple pre-defined grids with increasing difficulty.
    Movement rewards: -1 per step, +20 for palms, -100 for meteorites, +100 for goal.
    """
    metadata = {'render_modes': ['pygame'], 'render_fps': 5}
    

    
    def __init__(self, grid_list=None):
        """
        Initialize the environment.
        
        Args:
            grid_list (optional): List of custom grids. If None, uses pre-defined grids.
        """
        super().__init__()
        self.grid_size = 8
        self.cell_size = 100
        
        #If no grids are given, select the original ones
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
        
        #Pygame variables
        self.screen = None
        self.clock = None
        
        #Load images and scale them
        self.images = {
            'S': pygame.transform.scale(pygame.image.load("env_images/grass.png"), (self.cell_size, self.cell_size)),
            'M': pygame.transform.scale(pygame.image.load("env_images/meteorito.png"), (self.cell_size, self.cell_size)),
            'G': pygame.transform.scale(pygame.image.load("env_images/cueva.png"), (self.cell_size, self.cell_size)),
            'P': pygame.transform.scale(pygame.image.load("env_images/palmera.png"), (self.cell_size, self.cell_size))
        }
        self.agent_img = pygame.transform.scale(pygame.image.load("env_images/dino.png"), (self.cell_size, self.cell_size))
    




    #-----------------------------------------------#
    #---------------------GRIDS---------------------#
    #-----------------------------------------------#
    def _create_additional_grids(self):
        """
        Create additional grid layouts with increasing difficulty.
        
        Returns:
            list: List of 5 pre-defined 8x8 grids
        """

        grids = []
        
        #Grid 0
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

        #Grid 1
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
        
        
        #Grid 2
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


        #Grid 3
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

        #Grid 4
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
        """
        Set a specific grid as the current environment.
        
        Args:
            grid_idx (int): Index of the grid to use (0-4 for pre-defined grids)
        
        Notes:
            - Copies the selected grid and sets it as the current environment
            - Automatically finds and sets the agent's start position at 'S'
                - If no 'S' is found, starts at position (0,0)
        """
        self.current_grid_idx = grid_idx
        self.original_grid = self.grid_list[self.current_grid_idx].copy()
        self.grid = self.original_grid.copy()
        

        #Start when cell is 'S'
        start_positions = np.where(self.grid == 'S')
        if len(start_positions[0]) > 0:
            self.agent_pos = (start_positions[0][0], start_positions[1][0])
        else:
            self.agent_pos = (0, 0)  #If there is no 'S', start at (0,0)
    





    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options for reset
        
        Returns:
            tuple: (observation, info) where:
                - observation (int): Current state as integer (0-63)
                - info (dict): Empty dictionary for additional information
        
        Notes:
            - Resets the grid to its original state
            - Finds the start position marked with 'S'
            - Uses current grid, doesn't change to a different one
        """
        super().reset(seed=seed)

        #Restore current grid (don't change grid)
        self.grid = self.original_grid.copy()
        
        #Start when cell is 'S'
        start_positions = np.where(self.grid == 'S')
        if len(start_positions[0]) > 0:
            self.agent_pos = (start_positions[0][0], start_positions[1][0])
        else:
            self.agent_pos = (0, 0)  #If there is no 'S', start at (0,0)
        
        return self._get_state(), {}
    




    def _get_state(self):
        """
        Convert agent position to a single integer state.
        
        Returns:
            int: State representation from 0 to 63 (for 8x8 grid)
        
        Examples:
            Position (0,0) returns 0
            Position (0,1) returns 1
            Position (1,0) returns 8
        """
        i,j = self.agent_pos
        return i*self.grid_size + j
    




    def step(self, action):
        """
        Take a step in the environment by moving the agent.
        
        Args:
            action (int): Movement direction (0=up, 1=right, 2=down, 3=left)
                0: '↑',
                1: '→',
                2: '↓',
                3: '←' 
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info) where:
                - observation (int): New state after movement
                - reward (int): Reward for this step
                - terminated (bool): Whether episode ended
                - truncated (bool): Always False
                - info (dict): Empty dictionary
        
        Notes:
            - Palm trees disappear after being collected
            - Episode ends when reaching goal or meteorite
        """
        i,j = self.agent_pos
        if action == 0 and i>0: i -= 1
        if action == 1 and j<self.grid_size-1: j += 1
        if action == 2 and i<self.grid_size-1: i += 1
        if action == 3 and j>0: j -= 1
        self.agent_pos = (i,j)
        

        #-----------------------------------------------#
        #--------------------REWARDS--------------------#
        #-----------------------------------------------#
        cell = self.grid[i,j]
        reward, done = 0, False
        if cell == 'M': #Meteorite
            reward, done = -100, True
        elif cell == 'G': #Goal/Cave
            reward, done = 100, True
        elif cell == 'P': #Palm
            reward = 20 
            self.grid[i,j] = 'N'  
        else: #Nothing
            reward = -1  #Little negative reward when moving

        return self._get_state(), reward, done, False, {}
    


    def render(self, mode='pygame'):
        """
        Render the environment using Pygame visualization.
        
        Args:
            mode (str): Rendering mode (only 'pygame' supported)

        Notes:
            - Shows grid with different colors for each cell type
            - Displays images for special cells and agent
            - Window title shows current grid number
        """
        if mode != 'pygame':
            raise NotImplementedError("Solo modo pygame implementado")
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size*self.cell_size, self.grid_size*self.cell_size))
            pygame.display.set_caption(f"FrozenLake 8x8 - Grid {self.current_grid_idx+1}")
            self.clock = pygame.time.Clock()
        
        #Paint the grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size)
                cell_type = self.grid[i,j]
                
                #Backgroudnd color for each cell
                if cell_type == 'N':
                    pygame.draw.rect(self.screen, (114, 184, 68), rect) 
                elif cell_type == "M":
                    pygame.draw.rect(self.screen, (94, 170, 246), rect)
                elif cell_type == "P":
                    pygame.draw.rect(self.screen, (57, 119, 22), rect)
                else:
                    pygame.draw.rect(self.screen, (0,0,0), rect)

                #Put the image in each cell
                if cell_type in self.images:
                    self.screen.blit(self.images[cell_type], rect)
                
                pygame.draw.rect(self.screen, (255,255,255), rect, 1)  #Whithe border
        
        #Paint the agent
        ai,aj = self.agent_pos
        agent_rect = pygame.Rect(aj*self.cell_size, ai*self.cell_size, self.cell_size, self.cell_size)
        self.screen.blit(self.agent_img, agent_rect)
        
        pygame.display.flip()
        self.clock.tick(5)
    



    def close(self):
        """
        Close the environment and clean up Pygame resources.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None

