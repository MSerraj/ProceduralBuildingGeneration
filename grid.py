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
    # Stairs
    STAIRS = 20
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
                17: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                20: [[1, 1, 1], [1, 0, 1], [1, 1, 1]],}
        return grids.get(value)
    @staticmethod  
    def convert_to_3x3(grid):
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        new_grid = copy.deepcopy(grid)
        
        for i in range(rows):
            for j in range(cols):
                cell_value = grid[i][j]
                if grid[i][j] == 21:                        
                    north = grid[i-1][j] if i > 0 else -1
                    south = grid[i+1][j] if i < rows-1 else -1
                    east = grid[i][j+1] if j < cols-1 else -1
                    west = grid[i][j-1] if j > 0 else -1
                    
                    northwest = grid[i-1][j-1] if i > 0 and j > 0 else -1
                    northeast = grid[i-1][j+1] if i > 0 and j < cols-1 else -1
                    southwest = grid[i+1][j-1] if i < rows-1 and j > 0 else -1
                    southeast = grid[i+1][j+1] if i < rows-1 and j < cols-1 else -1
                    # Corners
                    if north != 21 and west != 21 and south == 21 and east == 21:
                        wall_type = Wall.DOWNRIGHT
                    elif north != 21 and east != 21 and south == 21 and west == 21:
                        wall_type = Wall.DOWNLEFT
                    elif south != 21 and west != 21 and north == 21 and east == 21:
                        wall_type = Wall.UPRIGHT
                    elif south != 21 and east != 21 and north == 21 and west == 21:
                        wall_type = Wall.UPLEFT
                        
                    # Horizontal corridor (east-west)
                    elif (west == 21 and east == 21) and ((north != 21 and south != 21) or \
                        (north == 21 and south != 21) or (north != 21 and south == 21)):
                        wall_type = Wall.HORIZ
                        
                    # Vertical corridor (north-south)
                    elif (north == 21 and south == 21) and ((west != 21 and east != 21) or \
                        (west == 21 and east != 21) or (west != 21 and east == 21)):
                        wall_type = Wall.VERT
                        
                    # Crossroads/intersection
                    elif north == 21 and south == 21 and east == 21 and west == 21:
                        wall_type = Wall.INSIDE
                        if northwest == 21 and northeast == 21 and southwest != 21 and southeast == 21:
                            wall_type = Wall.DOWNLEFT
                        elif northwest == 21 and northeast == 21 and southwest == 21 and southeast != 21:
                            wall_type = Wall.DOWNRIGHT
                        elif northwest != 21 and northeast == 21 and southwest == 21 and southeast == 21:
                            wall_type = Wall.UPLEFT
                        elif northwest == 21 and northeast != 21 and southwest == 21 and southeast == 21:
                            wall_type = Wall.UPRIGHT
                    # T-Junctions
                    elif north == 21 and south == 21 and (east == 21 or west == 21):
                        wall_type = Wall.T_UP if east == 21 else Wall.T_DOWN
                    elif east == 21 and west == 21 and (north == 21 or south == 21):
                        wall_type = Wall.T_LEFT if south == 21 else Wall.T_RIGHT

                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 21
                elif cell_value in {1, 18, 19}:
                    # Calculate bitmask using grid context
                    bitmask = 0
                    
                    if i > 0 and grid[i-1][j] in {1, 18, 19}: # up
                        bitmask += 1
                    if j < cols-1 and grid[i][j+1] in {1, 18, 19}: # right
                        bitmask += 2
                    if i < rows-1 and grid[i+1][j] in {1, 18, 19}: # down
                        bitmask += 4
                    if j > 0 and grid[i][j-1] in {1, 18, 19}: # left
                        bitmask += 8

                    # Find matching wall type
                    wall_type = next((w for w in Wall if w.value == bitmask), Wall.EMPTY)
                # Conversion of corridor into room 
                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 0
            
                elif cell_value == 0:
                    wall_type = Wall.EMPTY
                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 0
                elif cell_value == 17 or (128 <= cell_value and cell_value <= 144):
                    wall_type = Wall.INSIDE
                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 0
                else:
                    wall_type = Wall.EMPTY
                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 0
            
    
        return new_grid
    @staticmethod 
    def convert_corridor_to_room(grid):
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        new_grid = copy.deepcopy(grid)
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 21:                        
                    wall_type = Wall.EMPTY
                    north = grid[i-1][j] if i > 0 else -1
                    south = grid[i+1][j] if i < rows-1 else -1
                    east = grid[i][j+1] if j < cols-1 else -1
                    west = grid[i][j-1] if j > 0 else -1

                    # Corners
                    if north != 21 and west != 21 and south == 21 and east == 21:
                        wall_type = Wall.DOWNRIGHT
                    elif north != 21 and east != 21 and south == 21 and west == 21:
                        wall_type = Wall.DOWNLEFT
                    elif south != 21 and west != 21 and north == 21 and east == 21:
                        wall_type = Wall.UPRIGHT
                    elif south != 21 and east != 21 and north == 21 and west == 21:
                        wall_type = Wall.UPLEFT
                        
                    # Horizontal corridor (east-west)
                    elif (west == 21 and east == 21) and ((north != 21 and south != 21) or \
                        (north == 21 and south != 21) or (north != 21 and south == 21)):
                        wall_type = Wall.HORIZ
                        
                    # Vertical corridor (north-south)
                    elif (north == 21 and south == 21) and ((west != 21 and east != 21) or \
                        (west == 21 and east != 21) or (west != 21 and east == 21)):
                        wall_type = Wall.VERT
                        
                    # Crossroads/intersection
                    elif north == 21 and south == 21 and east == 21 and west == 21:
                        wall_type = Wall.INSIDE
                        
                    # T-Junctions
                    elif north == 21 and south == 21 and (east == 21 or west == 21):
                        wall_type = Wall.T_UP if east == 21 else Wall.T_DOWN
                    elif east == 21 and west == 21 and (north == 21 or south == 21):
                        wall_type = Wall.T_LEFT if south == 21 else Wall.T_RIGHT
                        
                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 21  # Keep as corridor if unhandled
                    
        return new_grid

    @staticmethod
    def from_wfc_grid(wfc_grid):
        """Convert WFC grid to our wall representation"""
        output = np.zeros((wfc_grid.height, wfc_grid.width), dtype=int)
        for y in range(wfc_grid.height):
            for x in range(wfc_grid.width):
                cell = wfc_grid.grid[y,x]
                if cell.collapsed and cell.options:
                    print(cell.options[0])
                    output[y,x] = cell.options[0].value
                else:
                    output[y,x] = 0  # Mark unprocessed cells
        return Wall.convert_to_3x3(output)

    def __str__(self):
        return f"{self.name}({self.value}, {self.ins})"
    
    def copy(self):
        """Copies self with deepcopy as well to avoid shared references"""
        new_copy = Wall(self.value)
        new_copy.ins = copy.deepcopy(self.ins)
        return new_copy

    def height_tot(self):
        return len(self.floors) * self.floor_h