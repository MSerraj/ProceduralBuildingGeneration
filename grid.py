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
                17: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],}
        return grids.get(value)

    def __str__(self):
        return f"{self.name}({self.value}, {self.ins})"
    
    def copy(self):
        """Copies self with deepcopy as well to avoid shared references"""
        new_copy = Wall(self.value)
        new_copy.ins = copy.deepcopy(self.ins)
        return new_copy


class Building:
    def __init__(self, base_floor):
        self.floors = [np.array(base_floor, dtype=int)]
        self.floor_h = 3  # meters
    
    def add_floor(self, new_floor):
        new_floor = np.array(new_floor, dtype=int)
        
        if new_floor.shape != self.floors[0].shape:# Check floor dimensions match
            raise ValueError("Floor dimensions don't make sense")
            
        if len(self.floors) > 0:# Check if new floor fits within previous floor's walls
            prev_floor = self.floors[-1]
            
            ext_mask = np.zeros_like(new_floor, dtype=bool)# Only validate exterior walls (edges of the grid)
            ext_mask[0, :] = True
            ext_mask[-1, :] = True   # Top edge, Bottom edge
            ext_mask[:, 0] = True 
            ext_mask[:, -1] = True   # Left edge, Right edge
            
            ext_new = new_floor[ext_mask]# Check exterior walls
            ext_prev = prev_floor[ext_mask]
            valid = np.all(ext_new <= ext_prev)
            
            if not valid:
                raise ValueError("New floor exceeds the one under")
        
        self.floors.append(new_floor)
        return self
    
    def height_tot(self):
        return len(self.floors) * self.floor_h