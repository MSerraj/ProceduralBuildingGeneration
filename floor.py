import numpy as np
import matplotlib.pyplot as plt
from utils import (plot_floorplan, visualize_grid, region_growing_simultaneous,
                   build_wall, generate_mapping_rectangles, find_rooms,
                   int_to_color, place_stairwell, find_optimal_corridor_tree)

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
        self.grid = find_optimal_corridor_tree(self.grid, min_width = min_width)
    
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