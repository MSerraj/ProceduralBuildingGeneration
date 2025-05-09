import numpy as np
import matplotlib.pyplot as plt
import random
import enum
import copy
from enum import Enum
from grid import *
from collections import defaultdict, deque
from utils import *


class FloorPlan:
    def __init__(self, grid, seeds):
        from wfc import WFCGrid

        """
        Initialize a FloorPlan.
        Args:
            grid (np.array): A 2D numpy array representing the floor plan.
        """
        self.grid = grid
        self.seeds = seeds
        self.original_grid = grid.copy()
        self.wfc_grid = WFCGrid(width=grid.shape[0], height=grid.shape[1])
    
    def grow_regions(self, rectangular=False):
        """
        Grow regions from seed points.
        Args:
            seeds (list): List of seed tuples [(x, y, value), ...].
        """
        if not rectangular:
            self.grid = region_growing_simultaneous(self.grid, self.seeds)
        else: 
            self.grid = region_growing_simultaneous_rectangular2(self.grid, self.seeds)
            #self.grid = remaining_pixels_rectangular(self.grid, self.seeds)
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
    
    def generate_stairs(self, x=2, y=2):
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
    
    def generate_wfc(self, max_attempts=100000000000):
        """Run the WFC algorithm to generate walls"""
        from wfc import WFCGrid
        attempt = 0
        while attempt < max_attempts:
            print(f"Attempt {attempt + 1}")
            try:
                self.wfc_grid = WFCGrid(width=self.grid.shape[0], height=self.grid.shape[1])
                while True:
                    cell = self.wfc_grid.get_lowest_entropy_cell()
                    if not cell:
                        break 
                    if not cell.options:
                        raise ValueError(f"Cell at {cell.position} has no options to collapse")
                    if cell.collapse():
                        self.wfc_grid.propagation_queue.append(cell)
                        self.wfc_grid.propagate_constraints()

                # Convert WFC result to grid
                wfc_enum = Wall.from_wfc_grid(self.wfc_grid)
                wfc_mask = wfc_enum

                new_grid = self.original_grid.copy()
                wall_values = {w.value for w in Wall}
                override = np.isin(wfc_mask, list(wall_values))
                new_grid[override] = wfc_mask[override]
                self.grid = new_grid
                return  # Success
            except ValueError as e:
                attempt += 1
                print(f"[Attempt {attempt}] Failed: {e}")
        raise RuntimeError("WFC failed after max_attempts")

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