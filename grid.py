import numpy as np
import matplotlib.pyplot as plt
import random
import enum
import copy
from enum import Enum
from scipy import ndimage
from collections import Counter


def place_windows(grid, window_value=5, window_size=8, min_spacing=4):
    window_size = min(window_size, 3)
    if window_size < 1:
        return np.copy(grid)
    
    new_grid = np.copy(grid)
    h, w = grid.shape
    
    # Identify exterior walls (1-valued only)
    outside_mask = (grid == 0)
    struct = ndimage.generate_binary_structure(2, 1)
    dilated_outside = ndimage.binary_dilation(outside_mask, structure=struct)
    exterior_walls = dilated_outside & (grid == 1)
    
    # Detect corners/thin walls (≥2 exterior neighbors)
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    exterior_neighbor_count = ndimage.convolve(
        outside_mask.astype(int), kernel, mode='constant', cval=0
    )
    invalid_mask = (exterior_neighbor_count >= 2)
    
    # Detect interior-side incidents (1,18,19 values)
    interior_bad_mask = np.zeros_like(exterior_walls, dtype=bool)
    for y in range(h):
        for x in range(w):
            if not exterior_walls[y, x]:
                continue
                
            for dy, dx in [(-1,0), (0,1), (1,0), (0,-1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    val = grid[ny, nx]
                    # Skip exterior (0) and exterior walls
                    if val == 0 or (val == 1 and exterior_walls[ny, nx]):
                        continue
                    # Flag if incident to 1,18,19 on interior side
                    if val in (1, 18, 19):
                        interior_bad_mask[y, x] = True
                        break
    
    # Create eligibility mask
    eligible = exterior_walls & ~invalid_mask & ~interior_bad_mask
    
    # Horizontal placement
    for y in range(h):
        segment_start = None
        for x in range(w + 1):  # +1 to flush last segment
            if x < w and eligible[y, x]:
                if segment_start is None:
                    segment_start = x
            elif segment_start is not None:
                segment_length = x - segment_start
                if segment_length >= window_size:
                    # For segments longer than 8 pixels, place two windows
                    if segment_length > 8:
                        # Calculate positions for two windows
                        total_needed = 2 * window_size + min_spacing
                        if segment_length >= total_needed:
                            start_x1 = segment_start + (segment_length - total_needed) // 2
                            start_x2 = start_x1 + window_size + min_spacing
                            
                            # Place first window
                            for i in range(window_size):
                                new_grid[y, start_x1 + i] = window_value
                            # Place second window
                            for i in range(window_size):
                                new_grid[y, start_x2 + i] = window_value
                        else:
                            # Fallback to single window if not enough space
                            start_x = segment_start + (segment_length - window_size) // 2
                            for i in range(window_size):
                                new_grid[y, start_x + i] = window_value
                    else:
                        # Place single window for segments ≤ 8 pixels
                        start_x = segment_start + (segment_length - window_size) // 2
                        for i in range(window_size):
                            new_grid[y, start_x + i] = window_value
                segment_start = None
    
    # Vertical placement
    for x in range(w):
        segment_start = None
        for y in range(h + 1):  # +1 to flush last segment
            if y < h and eligible[y, x]:
                if segment_start is None:
                    segment_start = y
            elif segment_start is not None:
                segment_length = y - segment_start
                if segment_length >= window_size:
                    # For segments longer than 8 pixels, place two windows
                    if segment_length > 8:
                        # Calculate positions for two windows
                        total_needed = 2 * window_size + min_spacing
                        if segment_length >= total_needed:
                            start_y1 = segment_start + (segment_length - total_needed) // 2
                            start_y2 = start_y1 + window_size + min_spacing
                            
                            # Place first window
                            for i in range(window_size):
                                new_grid[start_y1 + i, x] = window_value
                            # Place second window
                            for i in range(window_size):
                                new_grid[start_y2 + i, x] = window_value
                        else:
                            # Fallback to single window if not enough space
                            start_y = segment_start + (segment_length - window_size) // 2
                            for i in range(window_size):
                                new_grid[start_y + i, x] = window_value
                    else:
                        # Place single window for segments ≤ 8 pixels
                        start_y = segment_start + (segment_length - window_size) // 2
                        for i in range(window_size):
                            new_grid[start_y + i, x] = window_value
                segment_start = None

    return new_grid


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

    def __or__(self, a):
        return self.value | a.value
        
    def __and__(self, a):
        return self.value & a.value
    
    

    @staticmethod  
    def get_grid(value):
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
                20: [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                21: [[0,1,0],   [0,0,0],[0,1,0]], 
                26: [[0,0,0], [1,0,1], [0,0,0]],}
        return grids.get(value)
    @staticmethod  
    def convert_to_3x3(grid):
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        new_grid = copy.deepcopy(grid)
        room_grid = np.zeros(new_grid.shape)
        
        for i in range(rows):
            for j in range(cols):
                cell_value = grid[i][j]
                if grid[i][j] == 21:       
                    room_grid[i][j] = cell_value              
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
                        if north in {1, 18, 19} and west in {1, 18, 19}:
                            wall_type = Wall.CROSS
                        elif north in {1, 18, 19} and west not in {1, 18, 19}:
                            wall_type = Wall.T_RIGHT
                        elif north not in {1, 18, 19} and west in {1, 18, 19}:
                            wall_type = Wall.T_DOWN
                    elif north != 21 and east != 21 and south == 21 and west == 21:
                        wall_type = Wall.DOWNLEFT
                        if north in {1, 18, 19} and east in {1, 18, 19}:
                            wall_type = Wall.CROSS
                        elif north in {1, 18, 19} and east not in {1, 18, 19}:
                            wall_type = Wall.T_LEFT
                        elif north not in {1, 18, 19} and east in {1, 18, 19}:
                            wall_type = Wall.T_DOWN
                    elif south != 21 and west != 21 and north == 21 and east == 21:
                        wall_type = Wall.UPRIGHT
                        if south in {1, 18, 19} and west in {1, 18, 19}:
                            wall_type = Wall.CROSS
                        elif south in {1, 18, 19} and west not in {1, 18, 19}:
                            wall_type = Wall.T_RIGHT
                        elif south not in {1, 18, 19} and west in {1, 18, 19}:
                            wall_type = Wall.T_UP
                    elif south != 21 and east != 21 and north == 21 and west == 21:
                        wall_type = Wall.UPLEFT
                        if south in {1, 18, 19} and east in {1, 18, 19}:
                            wall_type = Wall.CROSS
                        elif south in {1, 18, 19} and east not in {1, 18, 19}:
                            wall_type = Wall.T_LEFT
                        elif south not in {1, 18, 19} and east in {1, 18, 19}:
                            wall_type = Wall.T_UP
                        
                    # Horizontal corridor (east-west)
                    elif (west == 21 and east == 21) and ((north != 21 and south != 21) or \
                        (north == 21 and south != 21) or (north != 21 and south == 21)):
                        wall_type = Wall.HORIZ
                        if (north == 21) and south in {1, 18, 19}:
                            wall_type = Wall.T_DOWN
                        if (north in {1, 18, 19}) and south == 21:
                            wall_type = Wall.T_UP
                        if (north in {1, 18, 19}) and south in {1, 18, 19}:
                            wall_type = Wall.CROSS
                    # Vertical corridor (north-south)
                    elif (north == 21 and south == 21) and ((west != 21 and east != 21) or \
                        (west == 21 and east != 21) or (west != 21 and east == 21)):
                        wall_type = Wall.VERT
                        if (west == 21) and east in {1, 18, 19}:
                            wall_type = Wall.T_RIGHT
                        if (west in {1, 18, 19}) and east == 21:
                            wall_type = Wall.T_LEFT
                        if (west in {1, 18, 19}) and east in {1, 18, 19}:
                            wall_type = Wall.CROSS
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

                    room_grid[i][j] = cell_value              
                    north = grid[i-1][j] if i > 0 else -1
                    south = grid[i+1][j] if i < rows-1 else -1
                    east = grid[i][j+1] if j < cols-1 else -1
                    west = grid[i][j-1] if j > 0 else -1
                    
                    northwest = grid[i-1][j-1] if i > 0 and j > 0 else -1
                    northeast = grid[i-1][j+1] if i > 0 and j < cols-1 else -1
                    southwest = grid[i+1][j-1] if i < rows-1 and j > 0 else -1
                    southeast = grid[i+1][j+1] if i < rows-1 and j < cols-1 else -1
            

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
                # Conversion of corridor into room 
                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 0
            
                elif cell_value == 0:
                    wall_type = Wall.EMPTY
                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 0
                elif cell_value == 17 or (128 <= cell_value and cell_value <= 144):
                    wall_type = Wall.INSIDE
                    room_grid[i][j] = cell_value
                    #print(f"new_cell {cell_value}")
                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 0
                else:
                    room_grid[i][j] = cell_value
                    #print(f"new_cell {cell_value}")
                    wall_type = Wall.EMPTY
                    new_grid[i][j] = wall_type.value if wall_type != Wall.EMPTY else 0
    
        return new_grid, room_grid
    
    @staticmethod 
    def convert_corridor_to_room(grid):
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        new_grid = copy.deepcopy(grid)
        
        for i in range(rows):
            for j in range(cols):
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
                    if north == 21 and west == 21 and south == 21 and east == 21 and \
                        northwest == 21 and northeast == 21 and southwest == 21 and southeast == 21:
                        wall_type_int = 17
                    else:
                        wall_type_int = 19
                        
                    new_grid[i][j] = wall_type_int
                    
        return new_grid

    @staticmethod
    def convert_3x3_to_3x3int(grid, room_grid):
        """Convert grid of Wall objects to full 3x3 integer grid"""
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        queue = []
        # Create output grid with 3x resolution
        new_grid = np.zeros((rows*3, cols*3), dtype=int)
        scaled_room_grid = np.kron(room_grid.astype(int).tolist(), np.ones((3, 3), dtype=int))
        for i in range(rows):
            for j in range(cols):                
                #print(f"position i={i}, j={j}, value={grid[i][j]}, and grid={cell_3x3}")
                y_start = i*3
                x_start = j*3

                cell_3x3 = Wall.get_grid(grid[i][j])
                cell_3x3 = np.array(cell_3x3)
                room_3x3 = scaled_room_grid[y_start:y_start+3, x_start:x_start+3]
                if np.all(cell_3x3 == 0) and not np.all(room_3x3 == 0):
                    new_grid[y_start:y_start+3, x_start:x_start+3] = room_3x3
                    room_val = int(room_3x3.flat[0])
                    if room_val not in {0, 1, 21} and room_val not in queue:
                        queue.append(room_val)
                else:
                    new_grid[y_start:y_start+3, x_start:x_start+3] = cell_3x3
                    if np.any(room_3x3 == 21):
                # Iterate over the 3x3 block relative to the current cell
                        for l_offset in range(3):
                            for m_offset in range(3):
                                if new_grid[y_start + l_offset, x_start + m_offset] == 0:
                                    new_grid[y_start + l_offset, x_start + m_offset] = 21
        return new_grid, queue
    
    @staticmethod
    def postprocess_3x3int(new_grid, queue):
        rows = len(new_grid)
        cols = len(new_grid[0]) if rows else 0     

        for i in range(rows):
            for j in range(cols):
                if new_grid[i][j] == 0 or new_grid[i][j] == 21:

                    north = new_grid[i-1][j]     if i > 0        else -1
                    south = new_grid[i+1][j]     if i < rows-1   else -1
                    east  = new_grid[i][j+1]     if j < cols-1   else -1
                    west  = new_grid[i][j-1]     if j > 0        else -1
                    nw = new_grid[i-1][j-1]      if i > 0 and j > 0           else -1
                    se = new_grid[i+1][j+1]      if i < rows-1 and j < cols-1 else -1
                    ne = new_grid[i-1][j+1]      if i > 0 and j < cols-1      else -1
                    sw = new_grid[i+1][j-1]      if i < rows-1 and j > 0      else -1

                    if north == 1 and south not in {0, 1, 21}:
                        new_grid[i][j] = south
                    elif south == 1 and north not in {0, 1, 21}:
                        new_grid[i][j] = north
                    elif east == 1 and west not in {0, 1, 21}:
                        new_grid[i][j] = west
                    elif west == 1 and east not in {0, 1, 21}:
                        new_grid[i][j] = east
                    elif nw == 1 and se not in {0, 1, 21}:
                        new_grid[i][j] = se
                    elif se == 1 and nw not in {0, 1, 21}:
                        new_grid[i][j] = nw
                    elif ne == 1 and sw not in {0, 1, 21}:
                        new_grid[i][j] = sw
                    elif sw == 1 and ne not in {0, 1, 21}:
                        new_grid[i][j] = ne
                    if new_grid[i][j] == 21 and any(n == 0 for n in [north, south, east, west, ne, sw, nw, se]):
                        new_grid[i][j] = 0
            rows, cols = new_grid.shape
        
        new_grid = place_windows(new_grid)

        for room_id in list(queue): # PLACE DOORS
            if room_id not in {0, 1, 21}:
                placed = False
                print(f"POOOGGGGG{queue}")
                room_mask = (new_grid == room_id)
                corridor_mask = (new_grid == 21)
                struct2 = ndimage.generate_binary_structure(2, 2)

                room_dilated = ndimage.binary_dilation(room_mask, structure=struct2).astype(np.uint8)
                corridor_dilated = ndimage.binary_dilation(corridor_mask, structure=struct2).astype(np.uint8)

                boundary = room_dilated & corridor_dilated  
                rows, cols = boundary.shape
                for i in range(rows):
                    for j in range(cols - 2):
                        if all(boundary[i, j + k] for k in range(5)) and not placed:
                            new_grid[i, j+1:j+4] = 4# Door assignment
                            placed = True
                    
                for i in range(rows - 2):
                    for j in range(cols):
                        if all(boundary[i+k, j] for k in range(5)) and not placed:
                            new_grid[i+1:i+4, j] = 4 # Door assignment
                            placed = True

                if not placed :
                    for i in range(rows):
                        for j in range(cols - 1):
                            if all(boundary[i, j + k] for k in range(3)) and not placed:
                                new_grid[i, j:j+3] = 4# Door assignment
                                placed = True
                    
                    for i in range(rows - 1):
                        for j in range(cols):
                            if all(boundary[i+k, j] for k in range(3)) and not placed:
                                new_grid[i:i+3, j] = 4 # Door assignment
                                placed = True
                if placed:
                    queue.remove(room_id)
            

        stairs_mask = (new_grid == 20)
        struct2     = ndimage.generate_binary_structure(2, 2)

        coords = np.argwhere(stairs_mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1 
        patch = new_grid[r0:r1, c0:c1]
        S = max(patch.shape)
        square_patch = np.full((S, S), 20, dtype=new_grid.dtype)

        new_grid[r0:r0+S, c0:c0+S] = square_patch

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
        return output

    def __str__(self):
        return f"{self.name}({self.value}, {self.ins})"
    
    def copy(self):
        """Copies self with deepcopy as well to avoid shared references"""
        new_copy = Wall(self.value)
        new_copy.ins = copy.deepcopy(self.ins)
        return new_copy

    def height_tot(self):
        return len(self.floors) * self.floor_h