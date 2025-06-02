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
        self.room_grid = np.zeros(grid.shape)
        self.int_grid_3x3 = []
        self.obj_grid_3x3 = []
        self.wfc_grid = WFCGrid(width=grid.shape[0], height=grid.shape[1])
    
    def set_grid(self, new_grid):
        self.grid = copy.deepcopy(new_grid)
    
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

    def go_to_3x3(self):
        #new_grid = Wall.convert_corridor_to_room(self.grid)
        new_grid, room_grid = Wall.convert_to_3x3(self.grid)
        self.grid = np.array(new_grid)
        self.room_grid = np.array(room_grid)
        #for row in self.grid:
        #    print(' '.join(map(str, row)))

    def corridor_into_room(self):
        new_grid = Wall.convert_corridor_to_room(self.grid)
        self.grid = np.array(new_grid)

    def go_to_3x3int(self):
        new_grid, queue = Wall.convert_3x3_to_3x3int(self.grid, self.room_grid)
        new_grid = Wall.postprocess_3x3int(new_grid, queue)
        self.grid = np.array(new_grid)
    
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
    
    def show(self, show_doors = False):
        """
        Display the floor plan using matplotlib.
        """
        plot_floorplan(self.color_coded(), seed_coordinates=None, save=False, show_doors=show_doors)

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
                    print(f"Cell: {cell} and its options {cell.options}")
                    if not cell.options:
                        raise ValueError(f"Cell at {cell.position} has no options to collapse")
                    if cell.collapse():
                        self.wfc_grid.propagation_queue.append(cell)
                        self.wfc_grid.propagate_constraints()
                output = np.zeros((self.wfc_grid.height, self.wfc_grid.width), dtype=int)
                for y in range(self.wfc_grid.height):
                    for x in range(self.wfc_grid.width):
                        cell = self.wfc_grid.grid[y,x]
                        if cell.collapsed and cell.options:
                            print(cell.options[0])
                            output[y,x] = cell.options[0].value
                        else:
                            output[y,x] = 255  # Mark unprocessed cells
                # Convert WFC result to grid
                self.wfc_grid = output
                print(f"Success! Grid: {self.wfc_grid}")
                self.grid = output
                return self.wfc_grid
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