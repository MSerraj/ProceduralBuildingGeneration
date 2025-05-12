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
                else:
                    new_grid[y_start:y_start+3, x_start:x_start+3] = cell_3x3
                    if np.any(room_3x3 == 21):
                # Iterate over the 3x3 block relative to the current cell
                        for l_offset in range(3):
                            for m_offset in range(3):
                                if new_grid[y_start + l_offset, x_start + m_offset] == 0:
                                    new_grid[y_start + l_offset, x_start + m_offset] = 21

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
                elif new_grid[i][j] == 1:
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
        return new_grid
    
    @staticmethod
    def place_doors(new_grid):
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

                    if north == 1 and south not in {1, 21}:
                        new_grid[i][j] = south
                    elif south == 1 and north not in {1, 21}:
                        new_grid[i][j] = north
                    elif east == 1 and west not in {1, 21}:
                        new_grid[i][j] = west
                    elif west == 1 and east not in {1, 21}:
                        new_grid[i][j] = east
                    elif nw == 1 and se not in {1, 21}:
                        new_grid[i][j] = se
                    elif se == 1 and nw not in {1, 21}:
                        new_grid[i][j] = nw
                    elif ne == 1 and sw not in {1, 21}:
                        new_grid[i][j] = sw
                    elif sw == 1 and ne not in {1, 21}:
                        new_grid[i][j] = ne
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
    
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
    
class FloorPlanVisualizer:
    def __init__(self, grid, cell_size=1.0):
        self.grid      = grid
        self.cell_size = cell_size
        self.rows      = len(grid)
        self.cols      = len(grid[0]) if self.rows else 0

        plt.style.use('seaborn-white')
        self.fig, self.ax = self._setup_plot()

        self.colors = {
            'ext_wall': '#2a2a2a',
            'int_wall': '#4a4a4a',
            'corridor': '#e8e4df',
            'door':     '#ffffff',
            'room_pal': ['#f7f4a8','#a8f7a4','#a4d4f7','#f7a4a4','#d4a4f7']
        }
        self.widths = {'ext':3.5, 'int':1.5}

    def _setup_plot(self):
        fig = plt.figure(figsize=(self.cols*0.6, self.rows*0.6))
        ax  = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlim(-0.5, self.cols-0.5)
        ax.set_ylim(self.rows-0.5, -0.5)
        ax.set_facecolor(self.colors['corridor'])
        ax.tick_params(axis='both', which='both', length=0)
        plt.xticks(np.arange(self.cols))
        plt.yticks(np.arange(self.rows))
        ax.grid(False)
        return fig, ax

    def _extract_walls(self):
        segs = {'ext_h':set(),'ext_v':set(),
                'int_h':set(),'int_v':set()}
        for i in range(self.rows):
            for j in range(self.cols):
                mask = self.grid[i][j].ins
                on_edge = i in (0,self.rows-1) or j in (0,self.cols-1)
                kind = 'ext' if on_edge else 'int'
                def st(ax,p1,p2):
                    segs[f"{kind}_{ax}"].add(tuple(sorted((p1,p2))))
                # top
                if all(mask[0][k]==1 for k in range(3)):  st('h',(j,i),(j+1,i))
                # bottom
                if all(mask[2][k]==1 for k in range(3)):  st('h',(j,i+1),(j+1,i+1))
                # left
                if all(mask[k][0]==1 for k in range(3)):  st('v',(j,i),(j,i+1))
                # right
                if all(mask[k][2]==1 for k in range(3)):  st('v',(j+1,i),(j+1,i+1))
        return segs

    def _draw_walls(self):
        segs = self._extract_walls()
        for kind, color, width in [
            ('ext','#2a2a2a',3.5), ('int','#4a4a4a',1.5)
        ]:
            lines = [ [p1,p2] 
                      for ax in ('h','v') 
                      for p1,p2 in segs[f"{kind}_{ax}"] ]
            lc = LineCollection(lines, colors=color,
                                linewidths=width, capstyle='round',
                                zorder=(4 if kind=='ext' else 3))
            self.ax.add_collection(lc)

    def _draw_doors(self):
        # white rectangles at door cells
        for i in range(self.rows):
            for j in range(self.cols):
                w = self.grid[i][j]
                if w in (Wall.HORIZ_DOOR, Wall.VERT_DOOR):
                    rect = Rectangle((j,i),1,1,
                                     facecolor=self.colors['door'],
                                     edgecolor='none',
                                     zorder=5)
                    self.ax.add_patch(rect)

    def _draw_rooms(self):
        visited = np.zeros((self.rows,self.cols),bool)
        palette = self.colors['room_pal']
        room_idx = 0

        for i in range(self.rows):
            for j in range(self.cols):
                if not visited[i,j] and self.grid[i][j]==Wall.INSIDE:
                    stack, cells = [(i,j)], []
                    while stack:
                        x,y = stack.pop()
                        if (0<=x<self.rows and 0<=y<self.cols 
                            and not visited[x,y]
                            and self.grid[x][y]==Wall.INSIDE):
                            visited[x,y]=True
                            cells.append((x,y))
                            stack += [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
                    if not cells: continue
                    rs = [c[0] for c in cells]; cs=[c[1] for c in cells]
                    minr,maxr,minc,maxc = min(rs),max(rs),min(cs),max(cs)
                    color = palette[room_idx % len(palette)]
                    room_idx+=1
                    rect = Rectangle((minc,minr),
                                     maxc-minc+1, maxr-minr+1,
                                     facecolor=color, alpha=0.4,
                                     edgecolor='none', zorder=1)
                    self.ax.add_patch(rect)

    def _draw_dimensions(self):
        # top
        self.ax.annotate('', xy=(self.cols,0), xytext=(0,0),
                         arrowprops=dict(arrowstyle='<->',color='gray'),
                         zorder=6)
        self.ax.text(self.cols/2,-0.2,f"{self.cols*self.cell_size:.1f}m",
                     ha='center',va='center',
                     backgroundcolor='white',fontsize=8,zorder=7)
        # left
        self.ax.annotate('', xy=(0,self.rows), xytext=(0,0),
                         arrowprops=dict(arrowstyle='<->',color='gray'),
                         zorder=6)
        self.ax.text(-0.2,self.rows/2,f"{self.rows*self.cell_size:.1f}m",
                     ha='center',va='center',rotation=90,
                     backgroundcolor='white',fontsize=8,zorder=7)

    def _draw_scale_bar(self):
        length_cells = 5/self.cell_size
        bar = Rectangle((1,self.rows-0.3),length_cells,0.15,
                        facecolor='black',edgecolor='none',zorder=8)
        self.ax.add_patch(bar)
        self.ax.text(1+length_cells+0.2,self.rows-0.2,
                     '5 m',ha='left',va='center',fontsize=8,zorder=9)

    def plot_blueprint(self):
        self._draw_rooms()
        self._draw_walls()
        self._draw_doors()
        self._draw_dimensions()
        self._draw_scale_bar()
        self.ax.set_title('Architectural Blueprint', pad=20, fontsize=14)
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    # --- example: border + two interior rooms + corridor + one door --- #
    R, C = 8, 10
    grid = [[Wall.EMPTY for _ in range(C)] for _ in range(R)]

    # perimeter solid walls
    for i in range(R):
        for j in range(C):
            if i in (0,R-1) or j in (0,C-1):
                grid[i][j] = Wall.CROSS

    # carve two rooms
    for i in range(2,5):
        for j in range(2,5):
            grid[i][j] = Wall.INSIDE

    for i in range(2,6):
        for j in range(6,9):
            grid[i][j] = Wall.INSIDE

    # place one horizontal door
    grid[4][5] = Wall.HORIZ_DOOR

    viz = FloorPlanVisualizer(grid, cell_size=1.0)
    viz.plot_blueprint()