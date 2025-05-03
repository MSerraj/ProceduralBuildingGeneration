import numpy as np
import matplotlib.pyplot as plt
import random
import enum
import copy
from enum import Enum


class Wall(Enum):
    # Floor
    EMPTY = 0 
    INSIDE = 17
    # Segments
    UP = 1 
    RIGHT = 2
    DOWN = 4
    LEFT = 8
    # Lengthwise
    HORIZ = 10  # 8 | 2
    VERT = 5    # 1 | 4
    # Corners
    UPLEFT = 9    # 1 | 8
    UPRIGHT = 3   # 1 | 2
    DOWNLEFT = 12 # 4 | 8
    DOWNRIGHT = 6 # 4 | 2
    # T-Junctions
    T_UP = 11     # 8 | 2 | 1
    T_DOWN = 14   # 8 | 4 | 2
    T_LEFT = 13   # 1 | 4 | 8
    T_RIGHT = 7   # 1 | 4 | 2
    # Cross Junction
    CROSS = 15    # 1 | 2 | 4 | 8
    # Doors
    HORIZ_DOOR = 26
    VERT_DOOR = 21
    
    def __init__(self, value):
        self._value_ = value
        self.ins = self.get_grid(value)

    def get_grid(self, value):
        """Returns the corresponding grid for a given value"""
        grids = {0: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 1: [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
                2: [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
                4: [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
                8: [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
                10: [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
                5: [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
                9: [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
                3: [[0, 1, 0], [0, 1, 1], [0, 0, 0]],
                12: [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
                6: [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
                11: [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
                14: [[0, 0, 0], [1, 1, 1], [0, 1, 0]],
                13: [[0, 1, 0], [1, 1, 0], [0, 1, 0]],
                7: [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
                15: [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                16: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # Single central pixel
                17: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],}
        return grids.get(value)
    @staticmethod  
    def convert_to_3x3(grid):
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        new_grid = copy.deepcopy(grid)
        
        for i in range(rows):
            for j in range(cols):
                cell_value = grid[i][j]
                if cell_value in {1, 21,18, 19}:
                    # Calculate bitmask using grid context
                    bitmask = 0
                    
                    if i > 0 and grid[i-1][j] in {1, 21, 18, 19}: # up
                        bitmask += 1
                    if j < cols-1 and grid[i][j+1] in {1, 21, 18, 19}: # right
                        bitmask += 2
                    if i < rows-1 and grid[i+1][j] in {1, 21, 18, 19}: # down
                        bitmask += 4
                    if j > 0 and grid[i][j-1] in {1, 21, 18, 19}: # left
                        bitmask += 8

                    # Find matching wall type
                    wall_type = next((w for w in Wall if w.value == bitmask), Wall.EMPTY)
                elif cell_value == 0:
                    wall_type = Wall.EMPTY
                elif cell_value == 17:
                    wall_type = Wall.INSIDE
                else:
                    wall_type = Wall.EMPTY

                new_grid[i][j] = wall_type
    
        return new_grid

    def __str__(self):
        return f"{self.name}({self.value}, {self.ins})"
    
    def copy(self):
        """Copies self with deepcopy as well to avoid shared references"""
        new_copy = Wall(self.value)
        new_copy.ins = copy.deepcopy(self.ins)
        return new_copy

    def height_tot(self):
        return len(self.floors) * self.floor_h