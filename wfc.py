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
    Wall.VERT, Wall.DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.T_DOWN,
    Wall.UP, Wall.UPLEFT, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT,
    Wall.HORIZ, Wall.CROSS,    
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
        min_entropy = 999999
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
                    if not neighbor.collapsed:
                        original_len = len(neighbor.options)
                        if cell.options:
                            neighbor.options = [
                                opt for opt in neighbor.options
                                if self.is_compatible(cell.options[0], opt, direction)
                            ]
                            if len(neighbor.options) < original_len:
                                self.propagation_queue.append(neighbor)
    def is_compatible(self, source, target, direction):
            # Define adjacency rules based on your bitmask requirements
            rules = {
                # Segments (single direction)
            Wall.UP: {
                'up': [Wall.VERT, Wall.DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.DOWNRIGHT, Wall.T_DOWN, Wall.CROSS,    ],
                'down': [Wall.EMPTY],
                'left': [Wall.EMPTY],
                'right': [Wall.EMPTY]
            },
            Wall.RIGHT: {
                'right': [Wall.HORIZ, Wall.T_LEFT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNLEFT, Wall.UPLEFT, Wall.CROSS,    ],
                'up': [Wall.EMPTY],
                'down': [Wall.EMPTY],
                'left': [Wall.EMPTY]
            },
            Wall.DOWN: {
                'down': [Wall.VERT, Wall.UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.UPLEFT, Wall.UPRIGHT, Wall.T_UP, Wall.CROSS,    ],
                'up': [Wall.EMPTY],
                'left': [Wall.EMPTY],
                'right': [Wall.EMPTY]
            },
            Wall.LEFT: {
                'left': [Wall.HORIZ, Wall.T_RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.CROSS,    ],
                'up': [Wall.EMPTY,  ],
                'down': [Wall.EMPTY,  ],
                'right': [Wall.EMPTY,  ]
            },
            # Lengthwise
            Wall.HORIZ: {
                'left': [Wall.HORIZ, Wall.T_RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.CROSS,    ],
                'right': [Wall.HORIZ, Wall.T_LEFT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNLEFT, Wall.UPLEFT, Wall.CROSS,    ],
                'up': [Wall.EMPTY],
                'down': [Wall.EMPTY]
            },
            Wall.VERT: {
                'up': [Wall.VERT, Wall.T_DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.DOWNRIGHT, Wall.CROSS,    ],
                'down': [Wall.VERT, Wall.T_UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.UPLEFT, Wall.UPRIGHT, Wall.CROSS,    ],
                'left': [Wall.EMPTY],
                'right': [Wall.EMPTY]
            },
            # Corners
            Wall.UPLEFT: {
                'up': [Wall.VERT, Wall.DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.DOWNRIGHT, Wall.T_DOWN, Wall.CROSS,    ],
                'left': [Wall.HORIZ, Wall.T_RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.CROSS,    ],
                'down': [Wall.EMPTY,  ],
                'right': [Wall.EMPTY,  ]
            },
            Wall.UPRIGHT: {
                'up': [Wall.VERT, Wall.DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.DOWNRIGHT, Wall.T_DOWN, Wall.CROSS,    ],
                'right': [Wall.HORIZ, Wall.T_LEFT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNLEFT, Wall.UPLEFT, Wall.CROSS,    ],
                'down': [Wall.EMPTY,  ],
                'left': [Wall.EMPTY,  ]
            },
            Wall.DOWNLEFT: {
                'down': [Wall.VERT, Wall.UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.UPLEFT, Wall.UPRIGHT, Wall.T_UP, Wall.CROSS,    ],
                'left': [Wall.HORIZ, Wall.T_RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.CROSS,    ],
                'up': [Wall.EMPTY,  ],
                'right': [Wall.EMPTY,  ]
            },
            Wall.DOWNRIGHT: {
                'down': [Wall.VERT, Wall.UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.UPLEFT, Wall.UPRIGHT, Wall.T_UP, Wall.CROSS,    ],
                'right': [Wall.HORIZ, Wall.T_LEFT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNLEFT, Wall.UPLEFT, Wall.CROSS,    ],
                'up': [Wall.EMPTY,  ],
                'left': [Wall.EMPTY,  ]
            },
            # T-Junctions
            Wall.T_UP: {
                'up': [Wall.VERT, Wall.DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.DOWNRIGHT, Wall.T_DOWN, Wall.CROSS,    ],
                'left': [Wall.HORIZ, Wall.T_RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.CROSS,    ],
                'right': [Wall.HORIZ, Wall.T_LEFT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNLEFT, Wall.UPLEFT, Wall.CROSS,    ],
                'down': [Wall.EMPTY,  ]
            },
            Wall.T_DOWN: {
                'down': [Wall.VERT, Wall.UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.UPLEFT, Wall.UPRIGHT, Wall.T_UP, Wall.CROSS,    ],
                'left': [Wall.HORIZ, Wall.T_RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.CROSS,    ],
                'right': [Wall.HORIZ, Wall.T_LEFT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNLEFT, Wall.UPLEFT, Wall.CROSS,    ],
                'up': [Wall.EMPTY,  ]
            },
            Wall.T_LEFT: {
                'left': [Wall.HORIZ, Wall.T_RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.CROSS,    ],
                'up': [Wall.VERT, Wall.DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.DOWNRIGHT, Wall.T_DOWN, Wall.CROSS,    ],
                'down': [Wall.VERT, Wall.UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.UPLEFT, Wall.UPRIGHT, Wall.T_UP, Wall.CROSS,    ],
                'right': [Wall.EMPTY,  ]
            },
            Wall.T_RIGHT: {
                'right': [Wall.HORIZ, Wall.T_LEFT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNLEFT, Wall.UPLEFT, Wall.CROSS,    ],
                'up': [Wall.VERT, Wall.DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.DOWNRIGHT, Wall.T_DOWN, Wall.CROSS,    ],
                'down': [Wall.VERT, Wall.UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.UPLEFT, Wall.UPRIGHT, Wall.T_UP, Wall.CROSS,    ],
                'left': [Wall.EMPTY,  ]
            },
            # Cross Junction
            Wall.CROSS: {
                'up': [Wall.VERT, Wall.DOWN, Wall.T_LEFT, Wall.T_RIGHT, Wall.DOWNLEFT, Wall.DOWNRIGHT, Wall.T_DOWN, Wall.CROSS,    ],
                'down': [Wall.VERT, Wall.UP, Wall.T_LEFT, Wall.T_RIGHT, Wall.UPLEFT, Wall.UPRIGHT, Wall.T_UP, Wall.CROSS,    ],
                'left': [Wall.HORIZ, Wall.T_RIGHT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNRIGHT, Wall.UPRIGHT, Wall.CROSS,    ],
                'right': [Wall.HORIZ, Wall.T_LEFT, Wall.T_DOWN, Wall.T_UP, Wall.DOWNLEFT, Wall.UPLEFT, Wall.CROSS,    ]
            }
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
            }

            
            # Get inverse direction for target
            inverse_dir = {'left':'right', 'right':'left', 'up':'down', 'down':'up'}[direction]
            return source in rules.get(target, {}).get(inverse_dir, [])
    