import numpy as np
import matplotlib.pyplot as plt
import random
import enum
import copy
from enum import Enum
from grid import *
from collections import defaultdict
from utils import (plot_floorplan, visualize_grid, region_growing_simultaneous,
                   build_wall, generate_mapping_rectangles, find_rooms,
                   int_to_color, place_stairwell, find_optimal_corridor_tree)

class WallMaker:
    def __init__(self, grid):
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.options = {}  # Stores possible values for each cell
        self.setup_options()
        
    def setup_options(self):
        """Set up possible wall/empty for blank spots"""
        # Inspired by Wave Function Collapse basics from:
        # https://github.com/mxgmn/WaveFunctionCollapse
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y,x] == 255 and self.near_wall(y,x):
                    self.options[(y,x)] = [1, 255]  # Could be wall or empty
                else:
                    self.options[(y,x)] = [self.grid[y,x]]  # Fixed value
                    
    def near_wall(self, y, x):
        """Check if cell is next to existing wall"""
        # Basic neighbor check from:
        # https://www.reddit.com/r/gamedev/comments/76l583/how_does_wave_function_collapse_work/
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.grid[ny,nx] == 1:
                        return True
        return False
    
    def make_walls(self, steps=200):
        """Main generation process"""
        # Core WFC logic adapted from simple examples:
        # https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/
        for _ in range(steps):
            # Find cells with most constraints
            uncertain = [pos for pos, opts in self.options.items() if len(opts) > 1]
            if not uncertain:
                break
            
            # Pick random uncertain cell (simpler than proper entropy)
            cell = random.choice(uncertain)
            
            # 75% chance to pick wall if possible
            if 1 in self.options[cell]:
                chosen = 1 if random.random() < 0.75 else 255
            else:
                chosen = random.choice(self.options[cell])
                
            self.grid[cell] = chosen
            self.update_neighbors(cell)
            
        return self.cleanup_walls()
    
    def update_neighbors(self, cell):
        """When we place a wall, make nearby cells more likely to be walls"""
        # Neighbor propagation idea from:
        # https://www.boristhebrave.com/2020/04/13/wave-function-collapse-explained/
        y, x = cell
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny = y + dy
                nx = x + dx
                if (ny, nx) in self.options:
                    if 1 in self.options[(ny, nx)]:
                        self.options[(ny, nx)].append(1)  # Increase wall chance
        
    def cleanup_walls(self):
        """Remove lonely walls"""
        # Common post-processing step mentioned in:
        # https://www.sidefx.com/tutorials/wfc-examples/
        clean_grid = self.grid.copy()
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if clean_grid[y,x] == 1:
                    neighbors = 0
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        if clean_grid[y+dy,x+dx] == 1:
                            neighbors += 1
                    if neighbors == 0:
                        clean_grid[y,x] = 255
        return clean_grid

class FloorPlan:
    def __init__(self, grid, seeds):
        """
        Initialize a FloorPlan.
        Args:
            grid (np.array): A 2D numpy array representing the floor plan.
        """
        self.grid = grid
        self.seeds = seeds
        self.original_grid = grid.copy()
    
    def grow_regions(self):
        """
        Grow regions from seed points.
        Args:
            seeds (list): List of seed tuples [(x, y, value), ...].
        """
        self.grid = region_growing_simultaneous(self.grid, self.seeds)
    
    def build_wall(self, a, b):
        """
        Build a straight wall between two points.
        Args:
            a (tuple): (x, y) start coordinate.
            b (tuple): (x, y) end coordinate.
        """
        self.grid = build_wall(self.grid, a, b)
    
    def generate_mapping_rectangles(self, rooms=(128, 129, 130, 132, 136, 144)):
        """
        Generate mapping rectangles for the given rooms.
        """
        self.grid = generate_mapping_rectangles(np.array(self.grid), rooms)
    
    def get_rooms(self):
        """
        Identify rooms from the grid.
        Returns:
            list: Each item is a tuple of (bounds, pixel values).
        """
        return find_rooms(self.grid)
    
    def generate_stairs(self, x=5, y=5):
        """
        Generate stairs in plan
        Args:
            x (int): (x) size.
            y (int): (y) size.
        """
        self.grid = place_stairwell(self.grid, x, y)

    def generate_corridors(self, min_width=4):
        """ 
        Generate corridors and widen them
        """
        self.grid = find_optimal_corridor_tree(self.grid)
    
    def color_coded(self):
        """
        Get a color-coded version of the current grid.
        Returns:
            np.array: 3-channel image array.
        """
        return int_to_color(self.grid)
    
    def show(self):
        """
        Display the floor plan using matplotlib.
        """
        plot_floorplan(self.color_coded(), seed_coordinates=self.seeds, save=False)

    def visualize(self):
        visualize_grid(self.grid, figsize=(16, 10), dpi=120, title="Floor Plan Visualization")
    
    def generate_organic_walls(self, iterations=500):
        """Generate natural-looking walls in unassigned areas"""
        generator = WallGenerator(self.grid)
        self.grid = generator.generate(iterations)
        # Clean up single pixel artifacts
        self.grid = self._remove_wall_isolates()
        
    def _remove_wall_isolates(self):
        """Remove single isolated wall pixels"""
        clean_grid = self.grid.copy()
        for i in range(1, self.grid.shape[0]-1):
            for j in range(1, self.grid.shape[1]-1):
                if self.grid[i,j] == 1:
                    neighbors = sum(self.grid[i+di,j+dj] == 1 
                                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)])
                    if neighbors == 0:
                        clean_grid[i,j] = 255
        return clean_grid