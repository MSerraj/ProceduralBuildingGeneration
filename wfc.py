# Add new file: wfc_core.py
import random
from grid import Wall
from utils import *
from floor import *
import cv2 as cv
from collections import defaultdict, deque
import os

class WFCell:
    def __init__(self, position):
        self.position = position
        self.collapsed = False
        self.options = self.options = list({
    Wall.EMPTY, Wall.CROSS,Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT,  Wall.T_DOWN,  Wall.VERT, Wall.DOWN, Wall.DOWNLEFT,
    Wall.UP, Wall.UPLEFT, Wall.DOWNRIGHT, Wall.UPRIGHT,
    Wall.HORIZ, 
})

    @property
    def entropy(self):
        return len(self.options) if not self.collapsed else 0
    
    def collapse(self):
        if self.collapsed or not self.options:
            return False
        self.collapsed = True
        self.options = [random.choice(self.options)]
        return True

class WFCGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.empty((height, width), dtype = object)
        for y in range(height):
            for x in range(width):
                self.grid[y, x] = WFCell((x, y))
        self.propagation_queue = deque()

    def get_lowest_entropy_cell(self):
        min_entropy = float('inf')
        candidates = []
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y,x]
                if not cell.collapsed:
                    if cell.entropy < min_entropy:
                        min_entropy = cell.entropy
                        candidates = [cell]
                    elif cell.entropy == min_entropy:
                        candidates.append(cell)
                        
        return random.choice(candidates) if candidates else None
    
    def propagate_constraints(self):
        while self.propagation_queue:
            cell = self.propagation_queue.popleft()
            x, y = cell.position
            if not cell.options:
                raise ValueError(f"Contradiction at {x}, {y} — cell has no options")
            for dx, dy, direction in [(-1,0,'left'), (1,0,'right'), 
                                    (0,-1,'up'), (0,1,'down')]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor = self.grid[ny,nx]
                    print(f"Is collapsed: {neighbor.collapsed} and options: {neighbor.options}")
                    if not neighbor.collapsed:
                        original_len = len(neighbor.options)
                        if cell.options:
                            neighbor.options = [
                                opt for opt in neighbor.options
                                if self.is_compatible(cell.options[0], opt, direction)
                            ]
                            if 0 < len(neighbor.options) < original_len:
                                self.propagation_queue.append(neighbor)
    def is_compatible(self, source, target, direction):
            # Define adjacency rules based on your bitmask requirements
            rules = {
                # Segments (single direction)
            "EMPTY": {
                'up': [Wall.EMPTY, ],#Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.LEFT, Wall.RIGHT],
                'down': [Wall.EMPTY, ],#Wall.LEFT, Wall.RIGHT, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN],
                'left': [Wall.EMPTY, ],#Wall.UP, Wall.DOWN, Wall.DOWNLEFT, Wall.UPLEFT],
                'right': [Wall.EMPTY, ],#Wall.UP, Wall.DOWN, Wall.DOWNRIGHT, Wall.UPRIGHT],
            },
            "CONNECTED": {
                'up': [Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT],
                'down': [Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT], 
                'left': [Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.T_RIGHT, Wall.CROSS, Wall.HORIZ],
                'right': [Wall.DOWNLEFT, Wall.UPLEFT, Wall.LEFT, Wall.T_DOWN, Wall.T_UP, Wall.T_LEFT, Wall.CROSS, Wall.HORIZ],
            },
            "UNCONNECTED": {
                'up': [Wall.EMPTY, Wall.UPRIGHT, Wall.UPLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_UP, Wall.HORIZ],
                'down': [Wall.EMPTY, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_DOWN, Wall.HORIZ],
                'left': [Wall.EMPTY, Wall.UPLEFT, Wall.DOWNLEFT, Wall.UP, Wall.DOWN, Wall.T_LEFT, Wall.VERT],
                'right': [Wall.EMPTY, Wall.UPRIGHT, Wall.DOWNRIGHT, Wall.UP, Wall.DOWN, Wall.T_RIGHT, Wall.VERT],
            },
            Wall.EMPTY: {
                'up': [Wall.EMPTY, Wall.UPRIGHT, Wall.UPLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_UP, Wall.HORIZ],
                'down': [Wall.EMPTY, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_DOWN, Wall.HORIZ],
                'left': [Wall.EMPTY, Wall.UPLEFT, Wall.DOWNLEFT, Wall.UP, Wall.DOWN, Wall.T_LEFT, Wall.VERT],
                'right': [Wall.EMPTY, Wall.UPRIGHT, Wall.DOWNRIGHT, Wall.UP, Wall.DOWN, Wall.T_RIGHT, Wall.VERT],
            },
            Wall.UP: {
                'up': [Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT],
                'down': [Wall.EMPTY, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_DOWN, Wall.HORIZ],
                'left': [Wall.EMPTY, Wall.UPLEFT, Wall.DOWNLEFT, Wall.UP, Wall.DOWN, Wall.T_LEFT, Wall.VERT],
                'right': [Wall.EMPTY, Wall.UPRIGHT, Wall.DOWNRIGHT, Wall.UP, Wall.DOWN, Wall.T_RIGHT, Wall.VERT],
            },
            Wall.RIGHT: {
                'right': [Wall.DOWNLEFT, Wall.UPLEFT, Wall.LEFT, Wall.T_DOWN, Wall.T_UP, Wall.T_LEFT, Wall.CROSS, Wall.HORIZ],
                'up': [Wall.EMPTY, Wall.UPRIGHT, Wall.UPLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_UP, Wall.HORIZ],
                'down': [Wall.EMPTY, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_DOWN, Wall.HORIZ],
                'left': [Wall.EMPTY, Wall.UPLEFT, Wall.DOWNLEFT, Wall.UP, Wall.DOWN, Wall.T_LEFT, Wall.VERT],
            },
            Wall.DOWN: {
                'down': [Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT], 
                'up': [Wall.EMPTY, Wall.UPRIGHT, Wall.UPLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_UP, Wall.HORIZ],
                'left': [Wall.EMPTY, Wall.UPLEFT, Wall.DOWNLEFT, Wall.UP, Wall.DOWN, Wall.T_LEFT, Wall.VERT],
                'right': [Wall.EMPTY, Wall.UPRIGHT, Wall.DOWNRIGHT, Wall.UP, Wall.DOWN, Wall.T_RIGHT, Wall.VERT],
            },
            Wall.LEFT: {
                'left': [Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.T_RIGHT, Wall.CROSS, Wall.HORIZ],
                'up': [Wall.EMPTY, Wall.UPRIGHT, Wall.UPLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_UP, Wall.HORIZ],
                'down': [Wall.EMPTY, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_DOWN, Wall.HORIZ],
                'right': [Wall.EMPTY, Wall.UPRIGHT, Wall.DOWNRIGHT, Wall.UP, Wall.DOWN, Wall.T_RIGHT, Wall.VERT],
            },
            # Lengthwise
            Wall.HORIZ: {
                'left': [Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.T_RIGHT, Wall.CROSS, Wall.HORIZ],
                'right': [Wall.DOWNLEFT, Wall.UPLEFT, Wall.LEFT, Wall.T_DOWN, Wall.T_UP, Wall.T_LEFT, Wall.CROSS, Wall.HORIZ],
                'up': [Wall.EMPTY, Wall.UPRIGHT, Wall.UPLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_UP,],# Wall.HORIZ],
                'down': [Wall.EMPTY, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_DOWN,],# Wall.HORIZ],
            },
            Wall.VERT: {
                'up': [Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT],
                'down': [Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT],
                'left': [Wall.EMPTY, Wall.UPLEFT, Wall.DOWNLEFT, Wall.UP, Wall.DOWN, Wall.T_LEFT,],# Wall.VERT],
                'right': [Wall.EMPTY, Wall.UPRIGHT, Wall.DOWNRIGHT, Wall.UP, Wall.DOWN, Wall.T_RIGHT,],# Wall.VERT],
            },
            # Corners
            Wall.UPLEFT: {
                'up': [Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT],
                'left': [Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.T_RIGHT, Wall.CROSS, Wall.HORIZ],
                'down': [Wall.EMPTY, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_DOWN, Wall.HORIZ],
                'right': [Wall.EMPTY, Wall.UPRIGHT, Wall.DOWNRIGHT, Wall.UP, Wall.DOWN, Wall.T_RIGHT, Wall.VERT],
            },
            Wall.UPRIGHT: {
                'up': [Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT],
                'right': [Wall.DOWNLEFT, Wall.UPLEFT, Wall.LEFT, Wall.T_DOWN, Wall.T_UP, Wall.T_LEFT, Wall.CROSS, Wall.HORIZ],
                'down': [Wall.EMPTY, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_DOWN, Wall.HORIZ],
                'left': [Wall.EMPTY, Wall.UPLEFT, Wall.DOWNLEFT, Wall.UP, Wall.DOWN, Wall.T_LEFT, Wall.VERT],
            },
            Wall.DOWNLEFT: {
                'down': [Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT], 
                'left': [Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.T_RIGHT, Wall.CROSS, Wall.HORIZ],
                'up': [Wall.EMPTY, Wall.UPRIGHT, Wall.UPLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_UP, Wall.HORIZ],
                'right': [Wall.EMPTY, Wall.UPRIGHT, Wall.DOWNRIGHT, Wall.UP, Wall.DOWN, Wall.T_RIGHT, Wall.VERT],
            },
            Wall.DOWNRIGHT: {
                'down': [Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT], 
                'right': [Wall.DOWNLEFT, Wall.UPLEFT, Wall.LEFT, Wall.T_DOWN, Wall.T_UP, Wall.T_LEFT, Wall.CROSS, Wall.HORIZ],
                'up': [Wall.EMPTY, Wall.UPRIGHT, Wall.UPLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_UP, Wall.HORIZ],
                'left': [Wall.EMPTY, Wall.UPLEFT, Wall.DOWNLEFT, Wall.UP, Wall.DOWN, Wall.T_LEFT, Wall.VERT],
            },
            Wall.CROSS: {
                'up': [Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT,  Wall.VERT],#Wall.CROSS,
                'down': [Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT,  Wall.VERT], #Wall.CROSS,
                'left': [Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.T_RIGHT,  Wall.HORIZ],#Wall.CROSS,
                'right': [Wall.DOWNLEFT, Wall.UPLEFT, Wall.LEFT, Wall.T_DOWN, Wall.T_UP, Wall.T_LEFT, Wall.HORIZ],#Wall.CROSS, 
            },
            # T-Junctions
            Wall.T_UP: {
                'up': [Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT],
                'left': [Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.T_RIGHT, Wall.CROSS, Wall.HORIZ],
                'right': [Wall.DOWNLEFT, Wall.UPLEFT, Wall.LEFT, Wall.T_DOWN, Wall.T_UP, Wall.T_LEFT, Wall.CROSS, Wall.HORIZ],
                'down': [Wall.EMPTY, Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_DOWN, Wall.HORIZ],
            },
            Wall.T_DOWN: {
                'down': [Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT], 
                'left': [Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.T_RIGHT, Wall.CROSS, Wall.HORIZ],
                'right': [Wall.DOWNLEFT, Wall.UPLEFT, Wall.LEFT, Wall.T_DOWN, Wall.T_UP, Wall.T_LEFT, Wall.CROSS, Wall.HORIZ],
                'up': [Wall.EMPTY, Wall.UPRIGHT, Wall.UPLEFT, Wall.LEFT, Wall.RIGHT, Wall.T_UP, Wall.HORIZ],
            },
            Wall.T_LEFT: {
                'left': [Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.T_RIGHT, Wall.CROSS, Wall.HORIZ],
                'up': [Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT],
                'down': [Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT], 
                'right': [Wall.EMPTY, Wall.UPRIGHT, Wall.DOWNRIGHT, Wall.UP, Wall.DOWN, Wall.T_RIGHT, Wall.VERT],
            },
            Wall.T_RIGHT: {
                'right': [Wall.DOWNLEFT, Wall.UPLEFT, Wall.LEFT, Wall.T_DOWN, Wall.T_UP, Wall.T_LEFT, Wall.CROSS, Wall.HORIZ],
                'up': [Wall.DOWNRIGHT, Wall.DOWNLEFT, Wall.DOWN, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT],
                'down': [Wall.UPRIGHT, Wall.UPLEFT, Wall.UP, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.CROSS, Wall.VERT], 
                'left': [Wall.EMPTY, Wall.UPLEFT, Wall.DOWNLEFT, Wall.UP, Wall.DOWN, Wall.T_LEFT, Wall.VERT],
            },
            
            }

            
            # Get inverse direction for target
            inverse_dir = {'left':'right', 'right':'left', 'up':'down', 'down':'up'}[direction]
            return source in rules.get(target, {}).get(inverse_dir, [])
    
            """
            
            # Cross Junction
            #
            # Doors
            ##   : {
            #    'left': [Wall.HORIZ,    , Wall.T_RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.CROSS],
            #    'right': [Wall.HORIZ,    , Wall.T_LEFT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNLEFT, Wall.UPLEFT, Wall.CROSS],
            #    'up': [Wall.EMPTY,  ],
            #    'down': [Wall.EMPTY,  ]
            #},
            #   : {
            #    'up': [Wall.VERT,    , Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.DOWNRIGHT, Wall.CROSS],
            #    'down': [Wall.VERT,    , Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.UPLEFT, Wall.UPRIGHT, Wall.CROSS],
            #    'left': [Wall.EMPTY,  ],
            #    'right': [Wall.EMPTY,  ]},
            #    """
    