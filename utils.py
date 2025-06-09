import numpy as np
from collections import deque, defaultdict, Counter
from math import sqrt
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.patches import Arc, Wedge, Patch
from matplotlib.colors import ListedColormap
from matplotlib import rcParams

# Constants
COLOR_TO_VALUE = {
    (0, 0, 0): 1,          # Walls (black)
    (255, 255, 255): 0,    # Outside (white)
    (255, 174, 201): 255,  # Inside (pink)
    (237, 28, 36): 128,    # Seed 1 (red)
    (0, 162, 232): 129,    # Seed 2 (blue)
    (34, 177, 76): 130,    # Seed 3 (green)
    (163, 73, 164): 132,   # Seed 4 (purple)
    (255, 127, 39): 136,   # Seed 5 (orange)
    (255, 242, 0): 144     # Seed 6 (yellow)
}

REVERSE_MAPPING = {
    18: (40, 81, 81),       # Rectangle wall
    19: (180, 180, 180),    # Separation between rooms
    20: (128, 64, 64),      # Escalator or stairs
    21: (128, 0, 64),       # Corridor
    128: (237, 28, 36),     # Seed 1 (red)
    129: (0, 162, 232),     # Seed 2 (blue)
    130: (34, 177, 76),     # Seed 3 (green)
    132: (163, 73, 164),    # Seed 4 (purple)
    136: (255, 127, 39),    # Seed 5 (orange)
    144: (255, 242, 0)      # Seed 6 (yellow)
}

NOT_ACCESS = {0, 1, 18, 19}  # Outside, hard wall, and soft wall

import numpy as np
from collections import deque, defaultdict

def image_to_int(floorplan):
    """
    Processes a floorplan image and converts it into a NumPy array with specific values.
    Also extracts the coordinates of seed pixels.
    Args:
        floorplan: Floorplan to be processed

    Returns:
        np.array: Processed NumPy array.
        list: List of seed coordinates in the format [(x1, y1, value1), (x2, y2, value2), ...].
    """
    color_to_value = {
        (0, 0, 0): 1,          # Walls (black)
        (255, 255, 255): 0,    # Outside (white)
        (255, 174, 201): 255,  # Inside (pink)
        #(237, 28, 36): 128,      # Seed 1 (red)
        (255, 0, 0): 128,       # Seed 1 (red)
        #(0, 162, 232): 129,      # Seed 2 (blue)
        (0, 0, 255): 129,      # Seed 2 (blue)
        #(34, 177, 76): 130,      # Seed 3 (green)
        (0, 255, 0): 130,      # Seed 3 (green)
        #(163, 73, 164): 132,    # Seed 4 (purple)
        (128, 0, 128): 132,    # Seed 4 (purple)
        (255, 128, 0): 136,    # Seed 5 (orange)
        #(255, 242, 0): 144     # Seed 6 (yellow)
        (255, 255, 0): 144     # Seed 6 (yellow)
    }
    y_max, x_max, _ = floorplan.shape
    floorplan_int = np.zeros((y_max, x_max), dtype=np.uint8)
    seeds = []

    for y in range(y_max):
        for x in range(x_max):
            pixel = tuple(floorplan[y, x])
            if pixel in color_to_value:
                floorplan_int[y, x] = color_to_value[pixel]
                if floorplan_int[y, x] in [128, 129, 130, 132, 136, 144]:
                    seeds.append((x, y, floorplan_int[y, x]))
            else:
                raise ValueError(f"Unexpected color {pixel} at position ({x}, {y})")

    return floorplan_int, seeds

def plot_floorplan(
    output_array,
    seed_coordinates=None,
    save=False,
    show_doors=False,
    corridor_tone=(204, 102, 178),
):
    """
    Plots the processed floorplan and highlights seed coordinates.

    Args:
        output_array (np.ndarray): H×W×3 RGB floorplan.
        seed_coordinates (List[Tuple[int,int,int]]): [(x, y, seed_value), …]
        save (bool): Whether to write a PNG to disk.
        show_doors (bool): If True, find door pixels and draw swing arcs.
        corridor_tone (Tuple[int,int,int]): RGB for corridor pixels.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(output_array, vmin=0, vmax=255)
    ax.set_title("Wave Function Collapse Floorplan", fontsize=40, fontweight="bold",x=0.5, y=1.02)
    ax.invert_yaxis()
    ax.axis("equal")
    
    # === 1) Draw doors and room legend if requested ===
    if show_doors:
        _draw_doors(ax, output_array, corridor_tone)
        _add_room_type_legend(ax)
    
    # === 2) Plot seeds with dedicated legend ===
    if seed_coordinates:
        _plot_seed_points(ax, seed_coordinates)
    
    # === 3) Finalize plot ===
    ax.axis("off")
    if save and seed_coordinates:
        first_x = seed_coordinates[0][0]
        plt.savefig(f"Floor_{first_x}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# --------------------- Helper Functions ---------------------
def _draw_doors(ax, output_array, corridor_tone):
    """Identifies door pixels and draws swing arcs for horizontal/vertical doors."""
    DOOR_COLOR = (128, 128, 64)
    ys, xs = np.where(np.all(output_array == DOOR_COLOR, axis=-1))
    
    # Process horizontal doors
    for y in np.unique(ys):
        row_xs = np.sort(xs[ys == y])
        runs = np.split(row_xs, np.where(np.diff(row_xs) != 1)[0] + 1)
        for run in runs:
            if 2 <= len(run) <= 4:
                _draw_door_wedge(ax, run, y, corridor_tone, output_array, horizontal=True)
    
    # Process vertical doors
    for x in np.unique(xs):
        col_ys = np.sort(ys[xs == x])
        runs = np.split(col_ys, np.where(np.diff(col_ys) != 1)[0] + 1)
        for run in runs:
            if 2 <= len(run) <= 4:
                _draw_door_wedge(ax, run, x, corridor_tone, output_array, horizontal=False)

def _draw_door_wedge(ax, run, fixed_coord, corridor_tone, output_array, horizontal):
    """Draws a door wedge based on orientation and adjacent space."""
    if horizontal:
        x_start, x_end = run[0], run[-1]
        radius = x_end - x_start + 2
        y_above = fixed_coord - 1
        if y_above >= 0:
            pixels_above = output_array[y_above, x_start : x_end + 1]
            corridor_mask = np.all(pixels_above == corridor_tone, axis=-1)
            stairs_mask = np.all(pixels_above == (128, 64, 64), axis=-1)
            above_free = np.any(~corridor_mask)
            stairs_above = np.any(~stairs_mask)
        else:
            above_free = True
            stairs_above = False
            
        if not above_free or not stairs_above:
            theta1, theta2 = 90, 180
            hinge_x, hinge_y = x_end + 1.5, fixed_coord - 0.5
        else:
            theta1, theta2 = 270, 360
            hinge_x, hinge_y = x_start - 0.5, fixed_coord + 0.5
    else:
        y_start, y_end = run[0], run[-1]
        radius = y_end - y_start + 2
        x_left = fixed_coord - 1
        if x_left >= 0:
            pixels_left = output_array[y_start : y_end + 1, x_left]
            corridor_mask = np.all(pixels_left == corridor_tone, axis=-1)
            stairs_mask = np.all(pixels_left == (128, 64, 64), axis=-1)
            left_free = np.any(~corridor_mask)
            stairs_free = np.any(~stairs_mask)
        else:
            left_free = True
            stairs_free = False
            
        if not left_free or not stairs_free:
            theta1, theta2 = 0, 90
            hinge_x, hinge_y = fixed_coord - 0.5, y_start - 0.5
        else:
            theta1, theta2 = 180, 270
            hinge_x, hinge_y = fixed_coord + 0.5, y_end + 1
    
    wedge = Wedge(
        (hinge_x, hinge_y), 
        r=radius, 
        theta1=theta1, 
        theta2=theta2,
        facecolor="gray", 
        edgecolor="none", 
        zorder=11
    )
    ax.add_patch(wedge)

def _add_room_type_legend(ax, rooms = (False, False, False, False, False, False)):
    """Adds a legend for room types outside the plot area."""
    REVERSE_MAPPING = {
        #"Door": (128, 128, 128),     # 4
        "Window": (128, 255, 255),    # Window
        "Wall": (0, 0, 0),      # Wall
        #"Room separator": (180, 180, 180),   # Room separator
        #"Stairs": (128, 64, 64),     # 20
        #"Corridor": (204, 102, 178),   # 21
        #"Unassigned": (255, 174, 201),
    }
    if rooms[0]:
        REVERSE_MAPPING["Seed A"] = (255, 153, 153)  # Seed 1
    if rooms[1]:
        REVERSE_MAPPING["Seed B"] = (153, 204, 255)  # Seed 2
    if rooms[2]:
        REVERSE_MAPPING["Seed C"] = (153, 255, 153)  # Seed 3
    if rooms[3]:
        REVERSE_MAPPING["Seed D"] = (204, 153, 255)  # Seed 1
    if rooms[4]:
        REVERSE_MAPPING["Seed E"] = (255, 204, 153)  # Seed 2
    if rooms[5]:
        REVERSE_MAPPING["Seed F"] = (255, 255, 170)  # Seed 3  
    patches = [
        Patch(
            facecolor=tuple(c/255 for c in color),
            edgecolor="black",
            label=str(room_type)
        ) 
        for room_type, color in REVERSE_MAPPING.items()
    ]
    ax.legend = ax.legend(
    handles=patches,
    loc="center left",
    bbox_to_anchor=(0.95, 0.5),
    frameon=True,
    title="Legend",
    fontsize=20,            # make label text 16 pt
    title_fontsize=23,      # make title text 18 pt
    markerscale=2.0,        # double the size of the patch/marker swatches
    labelspacing=0.5,       # increase vertical space between entries
    handlelength=3,         # length of the color‐patch box
    borderpad=1.0           # padding inside legend box
)

def _plot_seed_points(ax, seed_coordinates):
    """Plots seed points with labels and adds a legend."""
    seen_labels = set()
    for x, y, value in seed_coordinates:
        label = f"Seed {value}"
        ax.scatter(
            x, y,
            color="red",
            s=50,
            edgecolors="white",
            label=label if label not in seen_labels else "_nolegend_",
            zorder=12
        )
        ax.text(x, y, str(value), color="white", fontsize=12, 
               ha="center", va="center", zorder=13)
        seen_labels.add(label)
    
    ax.legend(loc="upper right", frameon=True, title="Seeds")

def region_growing_simultaneous(grid, seeds):
    """
    Grows seeds simultaneously by 1 pixel per round using 4-connected connectivity.
    Args:
        grid (np.array): Input floorplan grid.
        seeds (list): List of seed coordinates [(x, y, value), ...].
    Returns:
        np.array: Grid with grown regions.
    """
    result = grid.copy()
    # 4-connected movement directions (up, down, left, right)
    movement_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 8-connected directions for adjacency checks
    adjacent_directions = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),          (0, 1),
                           (1, -1),  (1, 0), (1, 1)]
    secondary_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-connected

    queue = deque()
    for x, y, value in seeds:
        # Store as (row, col, value)
        queue.append((y, x, value))
        result[y, x] = value

    while queue:
        round_size = len(queue)
        for _ in range(round_size):
            row, col, value = queue.popleft()
            
            # Attempt to grow in all 4 directions
            for dx, dy in movement_directions:
                new_row = row + dx
                new_col = col + dy
                
                # Check if new position is valid and unassigned
                if (0 <= new_row < grid.shape[0] and 
                    0 <= new_col < grid.shape[1] and 
                    result[new_row, new_col] == 255):
                    
                    # Check if adjacent to other regions (8-connected check)
                    has_adjacent_region = False
                    for adj_dx, adj_dy in adjacent_directions:
                        adj_row = new_row + adj_dx
                        adj_col = new_col + adj_dy
                        
                        if (0 <= adj_row < grid.shape[0] and 
                            0 <= adj_col < grid.shape[1]):
                            
                            cell_value = result[adj_row, adj_col]
                            if cell_value == 0 or (cell_value not in {255, value, 1}):
                                has_adjacent_region = True
                                break
                    
                    # Only grow if no adjacent regions found
                    if not has_adjacent_region:
                        result[new_row, new_col] = value
                        queue.append((new_row, new_col, value))
    
    return result

def region_growing_simultaneous_rectangular(grid, seeds):
    """
    Grows rectangular regions simultaneously by expanding all four sides each round
    until they hit obstacles or other regions.
    """
    result = grid.copy()
    # Convert seeds to rectangle representations (top, bottom, left, right, value)
    queue = deque()
    
    # Initialize queue with seed points as 1x1 rectangles
    for x, y, value in seeds:
        queue.append((y, y, x, x, value))  # (top, bottom, left, right, value)
        result[y, x] = value

    def can_expand(top, bottom, left, right, direction, value):
        """Check if a rectangle can expand in specified direction"""
        if direction == 'up':
            if top == 0: return False
            row = top - 1
            return np.all(result[row, left:right+1] == 255) and not has_adjacent(row, row+1, left, right, value)
                   
        elif direction == 'down':
            if bottom == result.shape[0]-1: return False
            row = bottom + 1
            return np.all(result[row, left:right+1] == 255) and not has_adjacent(row-1, row, left, right, value)
                   
        elif direction == 'left':
            if left == 0: return False
            col = left - 1
            return np.all(result[top:bottom+1, col] == 255) and not has_adjacent(top, bottom, col, col+1, value)
                   
        elif direction == 'right':
            if right == result.shape[1]-1: return False
            col = right + 1
            return np.all(result[top:bottom+1, col] == 255) and not has_adjacent(top, bottom, col-1, col, value)
                   
        return False

    def has_adjacent(t_start, t_end, l_start, l_end, value):
        """Check 8-connected adjacency for expansion area"""
        # Expand search area by 1 pixel in all directions
        t_start = max(0, t_start - 1)
        t_end = min(result.shape[0]-1, t_end + 1)
        l_start = max(0, l_start - 1)
        l_end = min(result.shape[1]-1, l_end + 1)
        
        region = result[t_start:t_end+1, l_start:l_end+1]
        return np.any((region != 255) & (region != value) & (region != 1))

    while queue:
        round_size = len(queue)
        expanded = set()

        for _ in range(round_size):
            top, bottom, left, right, value = queue.popleft()
            expanded = False

            # Try expanding in all four directions
            new_coords = []
            if can_expand(top, bottom, left, right, 'up', value):
                new_top = top - 1
                result[new_top, left:right+1] = value
                new_coords.append((new_top, bottom, left, right, value))
                expanded = True

            if can_expand(top, bottom, left, right, 'down', value):
                new_bottom = bottom + 1
                result[new_bottom, left:right+1] = value
                new_coords.append((top, new_bottom, left, right, value))
                expanded = True

            if can_expand(top, bottom, left, right, 'left', value):
                new_left = left - 1
                result[top:bottom+1, new_left] = value
                new_coords.append((top, bottom, new_left, right, value))
                expanded = True

            if can_expand(top, bottom, left, right, 'right', value):
                new_right = right + 1
                result[top:bottom+1, new_right] = value
                new_coords.append((top, bottom, left, new_right, value))
                expanded = True

            # If any expansion occurred, keep processing this rectangle
            if expanded:
                # Merge coordinates if multiple expansions
                merged = (
                    min(t for t,_,_,_,_ in new_coords),
                    max(b for _,b,_,_,_ in new_coords),
                    min(l for _,_,l,_,_ in new_coords),
                    max(r for _,_,_,r,_ in new_coords),
                    value
                )
                queue.append(merged)
            else:
                # Finalize this rectangle
                result[top:bottom+1, left:right+1] = value

    return result

def region_growing_simultaneous_rectangular2(grid, seeds):
    """
    Grows rectangular regions while handling partial edge obstacles.
    Expands directions independently when corners are blocked but direct path is clear.
    """
    result = grid.copy()
    queue = deque()

    # Initialize with seed points as rectangles
    for x, y, value in seeds:
        queue.append((y, y, x, x, value))  # (top, bottom, left, right, value)
        result[y, x] = value

    def can_expand_direction(top, bottom, left, right, direction, value):
        """Check expansion viability in specific direction with corner tolerance"""
        if direction == 'up':
            if top == 0: return False
            new_top = top - 1
            # Check only direct top cells, ignore diagonals
            return np.all(result[new_top, left:right+1] == 255) and \
                   not has_directional_conflict(new_top, new_top, left, right, value, 'up')
        
        # Similar checks for other directions
        elif direction == 'down':
            if bottom == result.shape[0]-1: return False
            new_bottom = bottom + 1
            return np.all(result[new_bottom, left:right+1] == 255) and \
                   not has_directional_conflict(new_bottom, new_bottom, left, right, value, 'down')
        
        elif direction == 'left':
            if left == 0: return False
            new_left = left - 1
            return np.all(result[top:bottom+1, new_left] == 255) and \
                   not has_directional_conflict(top, bottom, new_left, new_left, value, 'left')
        
        elif direction == 'right':
            if right == result.shape[1]-1: return False
            new_right = right + 1
            return np.all(result[top:bottom+1, new_right] == 255) and \
                   not has_directional_conflict(top, bottom, new_right, new_right, value, 'right')
        
        return False

    def has_directional_conflict(t_start, t_end, l_start, l_end, value, direction):
        """Check for conflicts only in expansion direction, ignoring diagonal corners"""
        # Define search area based on direction
        if direction == 'up':
            search_rows = [t_start]
            search_cols = range(l_start, l_end+1)
        elif direction == 'down':
            search_rows = [t_end]
            search_cols = range(l_start, l_end+1)
        elif direction == 'left':
            search_rows = range(t_start, t_end+1)
            search_cols = [l_start]
        elif direction == 'right':
            search_rows = range(t_start, t_end+1)
            search_cols = [l_end]

        # Check only direct path cells
        for r in search_rows:
            for c in search_cols:
                if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                    if result[r, c] not in {255, value, 1}:
                        return True
        return False

    while queue:
        round_size = len(queue)
        
        for _ in range(round_size):
            top, bottom, left, right, value = queue.popleft()
            expansions = []

            # Try expanding in all directions independently
            for direction in ['up', 'down', 'left', 'right']:
                if can_expand_direction(top, bottom, left, right, direction, value):
                    new_coords = expand_rectangle(
                        top, bottom, left, right, direction, value
                    )
                    expansions.append(new_coords)

            # Merge successful expansions
            if expansions:
                new_top = min(t for t, _, _, _, _ in expansions)
                new_bottom = max(b for _, b, _, _, _ in expansions)
                new_left = min(l for _, _, l, _, _ in expansions)
                new_right = max(r for _, _, _, r, _ in expansions)
                queue.append((new_top, new_bottom, new_left, new_right, value))
                
                # Update result for merged rectangle
                result[new_top:new_bottom+1, new_left:new_right+1] = value

    rows, cols = result.shape
    visited = np.zeros((rows, cols), dtype=bool)
    for y in range(rows):
        for x in range(cols):
            if result[y, x] == 255 and not visited[y, x]:
                # BFS to find connected component
                component = []
                queue = deque()
                queue.append((y, x))
                visited[y, x] = True
                while queue:
                    cy, cx = queue.popleft()
                    component.append((cy, cx))
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny = cy + dy
                        nx = cx + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            if result[ny, nx] == 255 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                # Determine adjacent region with most contact
                counts = defaultdict(int)
                for (cy, cx) in component:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny = cy + dy
                        nx = cx + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            val = result[ny, nx]
                            if val not in (255, 1):
                                counts[val] += 1
                if counts:
                    max_val = max(counts.items(), key=lambda x: x[1])[0]
                #else:
                #    max_val = 1  # Assign to wall if no adjacent regions
                # Update the component
                for (cy, cx) in component:
                    result[cy, cx] = max_val

    for y in range(rows):
        for x in range(cols):
            current_val = result[y, x]
            if current_val in {1, 19}: continue

            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    if result[ny, nx] != current_val and result[ny, nx] not in {1, 18, 19}:
                        result[y, x] = 19
                        break
                        
    # Fix loose corners in original walls (1s)

    def check_wall_ok(tile):
        y = tile[0]
        x = tile[1]
        center = result[y, x]
        bottom = result[y+1, x]
        bottomright = result[y+1, x+1]
        bottomleft = result[y+1, x-1]
        top = result[y-1, x]
        topright = result[y-1, x+1]
        topleft = result[y-1, x-1]
        right = result[y, x+1]
        left = result[y, x-1]
        return not right in {1, 18, 19} and bottom in {1, 18, 19} and bottomright in {1, 18, 19}
    
    for y in range(rows - 1):
        for x in range(cols - 1):
            if result[y, x] not in {0, 1, 18, 19}:
                center = result[y, x]
                bottom = result[y+1, x]
                bottomright = result[y+1, x+1]
                bottomleft = result[y+1, x-1]
                top = result[y-1, x]
                topright = result[y-1, x+1]
                topleft = result[y-1, x-1]
                right = result[y, x+1]
                left = result[y, x-1]
                neighbors = [bottom, top, right, left, bottomleft, bottomright,topright, topleft]
                cardinal = [top, bottom, left, right]
                diagonal = [topleft, topright, bottomleft, bottomright]

                wall_cardinal = len([n for n in cardinal if n in {1, 18, 19}])
                wall_diagonal = len([n for n in diagonal if n in {1, 18, 19}])

                # Fix loose corner: enough cardinal neighbors but no diagonal anchors

                if right in {1, 18, 19} and bottom in {1, 18, 19} and bottomright not in {1, 18, 19}:
                    if wall_cardinal <= 2 or result[y+1, x+1] == 0:
                        result[y, x] = 19
                    else:
                        result[y+1, x+1] = 19
                        

                if left in {1, 18, 19} and bottom in {1, 18, 19} and bottomleft not in {1, 18, 19}:
                    if wall_cardinal <= 2 or result[y+1, x+1] == 0:
                        result[y, x] = 19
                    else:
                        result[y+1, x-1] = 19

    return result

def expand_rectangle(top, bottom, left, right, direction, value):
    """Calculate new coordinates after expansion"""
    if direction == 'up':
        return (top-1, bottom, left, right, value)
    elif direction == 'down':
        return (top, bottom+1, left, right, value)
    elif direction == 'left':
        return (top, bottom, left-1, right, value)
    elif direction == 'right':
        return (top, bottom, left, right+1, value)


def assign_remaining_rectangles(grid):
    """
    Processes remaining 255 regions with strict rectangular handling:
    - Only assigns to directly adjacent rooms
    - Maintains contiguous blocks
    - Minimum 2x2 size requirement
    """
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    
    def get_contiguous_split(rect, neighbors):
        """Split rectangle into contiguous blocks based on adjacent rooms"""
        min_i, max_i, min_j, max_j = rect
        rooms = list(neighbors)
        
        # Single neighbor - full assignment
        if len(rooms) == 1:
            grid[min_i:max_i+1, min_j:max_j+1] = rooms[0]
            return
        
        # Two neighbors - simple split
        if len(rooms) == 2:
            if (max_j - min_j) > (max_i - min_i):  # Vertical split
                mid = min_j + (max_j - min_j) // 2
                grid[min_i:max_i+1, min_j:mid+1] = rooms[0]
                grid[min_i:max_i+1, mid+1:max_j+1] = rooms[1]
            else:  # Horizontal split
                mid = min_i + (max_i - min_i) // 2
                grid[min_i:mid+1, min_j:max_j+1] = rooms[0]
                grid[mid+1:max_i+1, min_j:max_j+1] = rooms[1]
            return
        
        # Three+ neighbors - quadrant assignment
        center_i = min_i + (max_i - min_i) // 2
        center_j = min_j + (max_j - min_j) // 2
        grid[min_i:center_i+1, min_j:center_j+1] = rooms[0]
        grid[min_i:center_i+1, center_j+1:max_j+1] = rooms[1]
        grid[center_i+1:max_i+1, min_j:center_j+1] = rooms[2 % len(rooms)]
        grid[center_i+1:max_i+1, center_j+1:max_j+1] = rooms[3 % len(rooms)]

    def find_rect_neighbors(rect):
        """Find adjacent rooms with direct contact (no diagonals)"""
        min_i, max_i, min_j, max_j = rect
        neighbors = set()
        
        # Check all four sides
        if min_i > 0: neighbors.update(grid[min_i-1, min_j:max_j+1])
        if max_i < h-1: neighbors.update(grid[max_i+1, min_j:max_j+1])
        if min_j > 0: neighbors.update(grid[min_i:max_i+1, min_j-1])
        if max_j < w-1: neighbors.update(grid[min_i:max_i+1, max_j+1])
        
        return {x for x in neighbors - {0, 1, 255} if x is not None}

    # Main processing loop
    for i in range(h):
        for j in range(w):
            if grid[i,j] == 255 and not visited[i,j]:
                # Find rectangular bounds
                q = deque([(i,j)])
                min_i = max_i = i
                min_j = max_j = j
                visited[i,j] = True
                
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x+dx, y+dy
                        if 0<=nx<h and 0<=ny<w and not visited[nx,ny] and grid[nx,ny]==255:
                            visited[nx,ny] = True
                            q.append((nx,ny))
                            min_i = min(min_i, nx)
                            max_i = max(max_i, nx)
                            min_j = min(min_j, ny)
                            max_j = max(max_j, ny)
                
                # Process only valid rectangles
                if (max_i - min_i >= 2) and (max_j - min_j >= 2):
                    neighbors = find_rect_neighbors((min_i, max_i, min_j, max_j))
                    if neighbors:
                        get_contiguous_split((min_i, max_i, min_j, max_j), neighbors)

    return grid


################# STAIRWELL ####################

def mark_corners_floor(grid):
    """Marks corners of the floor map (walls and boundaries)."""

    corners = []
    rows, cols = grid.shape
    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == 1:  # Wall
                if (x == 0 or x == rows - 1 or y == 0 or y == cols - 1) or (
                    (grid[x - 1][y] != 1 and grid[x][y - 1] != 1) or
                    (grid[x + 1][y] != 1 and grid[x][y - 1] != 1) or
                    (grid[x - 1][y] != 1 and grid[x][y + 1] != 1) or
                    (grid[x + 1][y] != 1 and grid[x][y + 1] != 1)):
                    corners.append((x, y))
    return corners

def place_stairwell(grid, size_x, size_y):
    h, w = grid.shape
    center = (h//2, w//2)
    corners = mark_corners_floor(grid)
    print(corners)
    for y, x in sorted(corners, key=lambda p: distance(p, center)):
        if try_place(grid, x, y, size_x, size_y):
            return grid
    return split_wall(grid, size_x, size_y)

import numpy as np

def try_place(grid, x, y, sx, sy):
    """Place stairwell (20) adjacent to outer corners with safe boundary checking."""
    # Check all 8 possible directions around the target coordinates
    directions = [
        (-sx, -sy), (-sx, 0), (-sx, sy),
        (0, -sy),          (0, sy),
        (sx, -sy), (sx, 0), (sx, sy)
    ]
    
    for dx, dy in directions:
        nx = x + dx
        ny = y + dy
        
        # Verify placement stays within grid boundaries
        if (0 <= nx < grid.shape[1] - sx + 1 and 
            0 <= ny < grid.shape[0] - sy + 1):
            
            # Extract candidate area
            area = grid[ny:ny+sy, nx:nx+sx]
            
            if np.all(area == 0):  # Only place in empty spaces
                adjacent_walls = 0
                grid_h, grid_w = grid.shape
                
                # Check four corners with boundary protection
                # Top-left adjacent
                if nx > 0 and ny > 0 and grid[ny-1, nx-1] == 1:
                    adjacent_walls += 1
                
                # Top-right adjacent
                if nx+sx < grid_w and ny > 0 and grid[ny-1, nx+sx] == 1:
                    adjacent_walls += 1
                
                # Bottom-left adjacent
                if nx > 0 and ny+sy < grid_h and grid[ny+sy, nx-1] == 1:
                    adjacent_walls += 1
                
                # Bottom-right adjacent
                if nx+sx < grid_w and ny+sy < grid_h and grid[ny+sy, nx+sx] == 1:
                    adjacent_walls += 1
                
                # Place if at least two corner-adjacent walls found
                if adjacent_walls >= 2:
                    # Place stairwell 1 unit away from walls if at edge
                    if nx == 0 or ny == 0 or nx+sx == grid_w or ny+sy == grid_h:
                        placement = grid[ny:ny+sy, nx:nx+sx]
                        if np.all(placement == 0):
                            grid[ny:ny+sy, nx:nx+sx] = 20
                            return True
                    else:
                        grid[ny:ny+sy, nx:nx+sx] = 20
                        return True
    return False

def split_wall(grid, sx, sy):
    """Emergency wall splitting"""
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y][x] == 1:
                mid_x = x - sx//2
                mid_y = y - sy//2
                if 0 <= mid_x < grid.shape[1]-sx and 0 <= mid_y < grid.shape[0]-sy:
                    grid[mid_y:mid_y+sy, mid_x:mid_x+sx] = 20
                    return grid
    return grid  # Fallback if no walls to split

def distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

############### CORRIDORS #######################

def get_neighbors(coord, skeleton):
    """Return 8-connected neighbors on the skeleton (only if value==1)."""
    y, x = coord
    neighbors = []
    """
    
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1), 
                   (1, -1),  (1, 0),  (0, 1)]:
    """
    for dy, dx in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
            if skeleton[ny, nx]:
                neighbors.append((ny, nx))
    return neighbors

def bfs_on_skeleton(skeleton, start_nodes, target_set):
    """
    Multi‐source BFS over `skeleton`, starting from `start_nodes`, until
    any pixel in `target_set` is reached. Returns a shortest path (list of (y,x) tuples).
    """
    required_hits=3
    # 1) Ensure all start_nodes are tuples
    start_nodes = [tuple(n) for n in start_nodes]

    # For each pixel, we'll remember “the best (maximum) number of target‐hits
    # seen so far on any path reaching that pixel.” If we reach the same pixel
    # later with a strictly higher hit‐count, we keep going from there.
    best_hits_at_pixel = {}   # maps (y,x) → int

    # parent_map[(pixel, hit_count)] = (prev_pixel, prev_hit_count)
    # so we can reconstruct the path later.
    parent_map = {}

    queue = deque()
    # Initialize BFS queue with each start‐node. If a start‐node is itself in
    # target_set, it already counts as one “hit.”
    for s in start_nodes:
        initial_hits = 1 if s in target_set else 0
        best_hits_at_pixel[s] = initial_hits
        parent_map[(s, initial_hits)] = None
        queue.append((s, initial_hits))

    while queue:
        current_pixel, hits_so_far = queue.popleft()
        # If we've already collected enough, we’re done!
        if hits_so_far >= required_hits:
            # Reconstruct the path (just the sequence of pixels) for this state.
            path = []
            state = (current_pixel, hits_so_far)
            while state is not None:
                pix, h = state
                path.append(pix)
                state = parent_map[state]
            path.reverse()
            return path

        # Otherwise, explore neighbors
        for nb in get_neighbors(current_pixel, skeleton):
            nb = tuple(nb)  # force to tuple so it’s hashable

            # Determine how many “hits” we’d have if we step into nb:
            #   - if nb is in target_set AND we have not already counted it,
            #     increment hits_so_far by 1.
            #   - BUT—because BFS never revisits the SAME pixel twice on a SINGLE path,
            #     we only ever count each pixel once. So simply:
            new_hits = hits_so_far + (1 if nb in target_set else 0)

            # Have we ever reached nb before with >= new_hits? If so, skip.
            prev_best = best_hits_at_pixel.get(nb, -1)
            if new_hits <= prev_best:
                # That means we’ve already reached nb via some path
                # that gathered as many or more target‐hits—no need to revisit.
                continue

            # Otherwise, this is a strictly better “hit‐count” at nb.
            best_hits_at_pixel[nb] = new_hits
            parent_map[(nb, new_hits)] = (current_pixel, hits_so_far)
            queue.append((nb, new_hits))

    # If queue empties without ever collecting `required_hits` hits, return None.
    return None

def find_room_loose(grid):
    """
    Returns all boundary pixels of `rooms` adjacent.
    """
    rows, cols = grid.shape
    for y in range(rows):
        for x in range(cols):
            center = grid[y, x]
            if center in {1, 18, 19}:
                def get(y_, x_):
                    return grid[y_, x_] if 0 <= y_ < rows and 0 <= x_ < cols else -1  # use -1 for out-of-bounds

                bottom      = get(y+1, x)
                bottomright = get(y+1, x+1)
                bottomleft  = get(y+1, x-1)
                top         = get(y-1, x)
                topright    = get(y-1, x+1)
                topleft     = get(y-1, x-1)
                right       = get(y, x+1)
                left        = get(y, x-1)
                neighbors = [bottom, top, right, left, bottomleft, bottomright,topright, topleft]
                cardinal = [top, bottom, left, right]
                diagonal = [topleft, topright, bottomleft, bottomright]

                wall = [n for n in neighbors if n in {0, 1, 18, 19}]
                ziro = [n for n in neighbors if n in {0}]
                corridor = [n for n in neighbors if n in {20, 21}]
                room_pix = Counter(wall).most_common(1)
                room_pix = room_pix[0][0]

                wall_cardinal = len([n for n in cardinal if n in {1, 18, 19, 21}])
                wall_diagonal = len([n for n in diagonal if n in {1, 18, 19, 21}])

                #if len(corridor) >= 1:
                #    grid[y, x] = room_pix
                if room_pix == 0 and len(ziro) >= 3 and len(corridor) >= 3:
                    print("CORNERRRRRRRRRRRRRRRRrrrr")
                    grid[y, x] = 0
                elif room_pix == 0 and len(ziro) >= 5 and len(corridor) >= 2:
                    grid[y, x] = 0

    return grid



import numpy as np
from scipy import ndimage

def find_room_boundaries(grid, room_id, skeleton):
    """
    Returns all boundary pixels of `room_id` adjacent to the skeleton.
    """
    room_mask = (grid == room_id)
    struct2 = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(room_mask, structure=struct2).astype(room_mask.dtype)
    boundary = dilated & (skeleton == 1)  # Overlap with skeleton
    return np.argwhere(boundary)

def find_optimal_corridor_tree(grid, min_width = 4, through_room = None, min_adjacency=4):
    """
    Constructs a minimal corridor tree (corridor pixels marked as 21)
    that connects the stairwell (20) to all rooms (all values except 0,1,18,19,20,21).
    The algorithm uses the skeleton of possible corridor locations and a
    multi-target BFS to add the shortest connection from the growing tree
    to each unconnected room.
    """
    skeleton = np.where((grid == 1) | (grid == 18) | (grid == 19) | (grid == through_room), 1, 0).astype(np.uint8)

    # Rooms to connect (exclude through_room from target list)
    rooms = [val for val in np.unique(grid)  if val not in {0, 1, 18, 19, 21, through_room} and val != 20]
    
    stair_pos = np.argwhere(grid == 20)
    if len(stair_pos) == 0:
        raise ValueError("No stairwell (20) found in grid")
    
    boundaries = {}
    for room in rooms + [20]:
        boundaries[room] = find_room_boundaries(grid, room, skeleton)
    #print(tuple([boundaries[20][0]]).dtype)
    # --- Initialization ---
    # Start with a single seed from the stairwell boundary.
    if len(boundaries[20]) == 0:
        raise ValueError("Stairwell has no valid boundary on the skeleton")
    
    tree_nodes = set([tuple(boundaries[20][0])])  # our initial tree (a single pixel)
    connected_rooms = {20}  # the stairwell is our seed
    unconnected_rooms = set(rooms)  # remaining rooms
    # Build mapping for each room: room id -> set of boundary pixels (as candidates)
    target_boundaries = {room: {tuple(coord) for coord in boundaries[room]} for room in unconnected_rooms}
    #tuple(map(tuple, arr))
    # --- Iteratively add the shortest connection from the tree to a new room ---
    while unconnected_rooms:
        # Create a union of all target pixels from unconnected rooms.
        target_union = {}
        for room, nodes in target_boundaries.items():
            for node in nodes:
                target_union[node] = room  # if a pixel belongs to multiple, one mapping is enough.
        
        # Run BFS from the current tree_nodes until we hit any target boundary pixel.
        path = bfs_on_skeleton(skeleton, tree_nodes, set(target_union.keys()))
        if path is None:
            break
            raise ValueError("No path found from the current tree to one of the rooms")
        
        # The reached target pixel tells us which room is connected.
        target_pixel = path[-1]
        room_id = target_union[target_pixel]
        
        # Add the entire path to the tree.
        for p in path:
            tree_nodes.add(p)
        
        # Mark the room as connected.
        connected_rooms.add(room_id)
        unconnected_rooms.remove(room_id)
        del target_boundaries[room_id]
    
    # --- Annotate the corridors in the grid ---
    for y, x in tree_nodes:
        if grid[y, x] not in {through_room, 20}:  # Do not overwrite the through_room
            grid[y, x] = 21

    grid = widen_corridors(grid)
    grid = find_room_loose(grid)
    return grid

def widen_corridors(grid, val=21, iterations=2):
    """ 
    Expands corridors to minimum width using morphological dilation, 
    while preserving existing walls (non-zero values).
    """
    grid = np.array(grid, dtype=np.uint8).copy()

    # Binary mask of corridors and empty space
    corridor_mask = (grid == val).astype(np.uint8)
    empty_mask = (grid == 0).astype(np.uint8)

    # Identify corridor pixels adjacent to empty space
    # Dilate empty_mask by 1 to mark neighbors
    empty_dilated = cv.dilate(empty_mask, cv.getStructuringElement(cv.MORPH_RECT, (3,3)))
    adjacent_to_empty = corridor_mask & (empty_dilated > 0)

    # Kernel definitions
    k2 = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    k3 = cv.getStructuringElement(cv.MORPH_RECT, (3,3))

    # Dilate separately
    dilated_adj = cv.dilate(adjacent_to_empty, k3, iterations=iterations)
    dilated_other = cv.dilate(corridor_mask , k2, iterations=iterations)

    # Combine and expand only into empty spaces
    dilated = (dilated_adj | dilated_other).astype(bool)
    expand_mask = dilated & (grid != 0) & (grid != 20)

    # Set expanded areas to corridor value
    grid[expand_mask] = val
    return grid



######################### Plotting #################################3


def int_to_color(result):

    """Converts integer grid values to RGB colors."""

    reverse_mapping = {
        4: (128, 128, 64),     # Door
        5: (128, 255, 255),    # Window
        18: (40, 81, 81),       # Wall color
        19: (180, 180, 180),    # Room separator
        20: (128, 64, 64),      # Escalator/stairs
        21: (204, 102, 178),       # Corridor
        255: (255, 174, 201),
        #128: (237, 28, 36),     # Seed 1 (red)
        #128: (255, 0, 0),       # Seed 1 (red)
        #129: (0, 162, 232),     # Seed 2 (blue)
        #129: (0, 128, 255),     # Seed 2 (blue)
        #130: (34, 177, 76),     # Seed 3 (green)
        #130: (0, 255, 0),     # Seed 3 (green)
        #132: (163, 73, 164),    # Seed 4 (purple)
        #132: (128, 0, 128),    # Seed 4 (purple)
        #136: (255, 127, 39),    # Seed 5 (orange)
        #144: (255, 242, 0)      # Seed 6 (yellow)
        #144: (255, 255, 0)      # Seed 6 (yellow)

            # Soft Pastel Seed Colors (moderate pastel)
        128: (255, 153, 153),   # Seed 1 (light pastel red)
        129: (153, 204, 255),   # Seed 2 (light pastel blue)
        130: (153, 255, 153),   # Seed 3 (light pastel green)
        132: (204, 153, 255),   # Seed 4 (light pastel purple)
        136: (255, 204, 153),   # Seed 5 (light pastel orange)
        144: (255, 255, 170)    # Seed 6 (light pastel yellow)
    }
    color_coded_grid = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            value = result[y, x]
            if value in reverse_mapping:
                color_coded_grid[y, x] = reverse_mapping[value]
            elif value == 1:
                color_coded_grid[y, x] = (0, 0, 0)  # Wall (black)
            elif value == 0:
                color_coded_grid[y, x] = (255, 255, 255)  # Outside (white)
            elif value == 255:
                color_coded_grid[y, x] = (255, 192, 203)  # Unassigned (pink)
    return color_coded_grid



"""
######################### Room straightening #################################3


def fit_largest_rectangle(grid, room_number):
    
    Fit largest rectangle in room with room cells.
    Return the coordinates of the largest rectangle as (top, left, bottom, right).
    
    if grid.size == 0:
        return None
    
    max_area = 0
    max_coords = (0, 0, 0, 0)
    rows, cols = grid.shape
    heights = [0] * cols  # Initialize histogram heights
    
    for row_idx in range(rows):
        for col in range(cols):
            if grid[row_idx][col] == room_number:
                heights[col] += 1
            else:
                heights[col] = 0
        
        # Use a stack to find largest rectangle in histogram
        stack = [-1]
        for i in range(cols + 1):
            current_height = heights[i] if i < cols else 0
            while stack[-1] != -1 and current_height < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    left = stack[-1] + 1
                    right = i - 1
                    top = row_idx - h + 1
                    bottom = row_idx
                    max_coords = (top, left, bottom, right)
            stack.append(i)
    
    return max_coords if max_area > 0 else None

def build_wall(grid, a, b):
    
    Builds a straight wall between points a and b on the grid.
    Walls can be either horizontal or vertical.
    
    x_a, y_a = a
    x_b, y_b = b

    if not (x_a == x_b or y_a == y_b):
        raise ValueError("Wall must be horizontal or vertical")
        
    if y_a == y_b:  # Horizontal wall
        for x in range(min(x_a, x_b), max(x_a, x_b) + 1):
            if grid[x][y_a] in (1, 0):
                continue
            grid[x][y_a] = 18  # Mark as wall

    else:  # Vertical wall
        for y in range(min(y_a, y_b), max(y_a, y_b) + 1):
            if grid[x_a][y] in (1, 0):
                continue
            grid[x_a][y] = 18  # Mark as wall

    return grid

def generate_mapping_rectangles(grid, rooms=(128, 129, 130, 132, 136, 144)):
    
    For each room number, find its largest rectangle and build walls around it.
    
    # Step 1: Find corners from floor and rooms
    corners_rooms = []
    corners_T = []
    corners_floor = mark_corners_floor(grid)

    for room in rooms:
        rect = fit_largest_rectangle(grid, room)
        if rect is None:
            continue

        top, left, bottom, right = rect
        # Top-left
        tl = (top, left)
        corners_rooms.append(tl)
        corners_T.extend(mark_corners_T(grid, *tl, 'top_left'))

        # Bottom-left
        bl = (bottom, left)
        corners_rooms.append(bl)
        corners_T.extend(mark_corners_T(grid, *bl, 'bottom_left'))

        # Top-right
        tr = (top, right)
        corners_rooms.append(tr)
        corners_T.extend(mark_corners_T(grid, *tr, 'top_right'))

        # Bottom-right
        br = (bottom, right)
        corners_rooms.append(br)
        corners_T.extend(mark_corners_T(grid, *br, 'bottom_right'))
        
    corners_all = list(set(corners_floor + corners_rooms + corners_T))

    # Step 2: Generate rectangles from corners
    corners_all = gen_rect_corners(corners_all)
    
    for rect in corners_all:
        (x1, y1), (x2, y2) = rect
        grid = build_wall(grid, (x1, y1), (x2, y1))  # Top wall
        grid = build_wall(grid, (x1, y2), (x2, y2))  # Bottom wall
        grid = build_wall(grid, (x1, y1), (x1, y2))  # Left wall
        grid = build_wall(grid, (x2, y1), (x2, y2))  # Right wall
    #plot_floorplan(int_to_color(grid), save=False)
    grid = fill_rooms_with_dominant_color(grid)
    grid = replace_walls(grid)
    grid = replace_walls(grid)
    grid = replace_walls(grid)
    return grid

def find_rooms(grid):
    Identify rooms using flood fill, separated by walls (18).
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    rooms = []

    def bfs(start_x, start_y):
        queue = deque([(start_x, start_y)])
        visited[start_x][start_y] = True
        pixels = []
        min_x = max_x = start_x
        min_y = max_y = start_y

        while queue:
            x, y = queue.popleft()
            pixels.append(grid[x][y])
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if not visited[nx][ny] and grid[nx][ny] != 18 and grid[nx][ny] != 1 and grid[nx][ny] != 0:
                        visited[nx][ny] = True
                        queue.append((nx, ny))
        return ((min_x, min_y), (max_x, max_y)), pixels

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and grid[i][j] not in (18, 1):
                bounds, pixels = bfs(i, j)
                rooms.append((bounds, pixels))
    
    return rooms

def fill_rooms_with_dominant_color(grid):
    Fill each room with its dominant color (excluding walls).
    rooms = find_rooms(grid)
    new_grid = np.copy(grid)
    
    # Ensure wall boundaries are preserved
    for (min_x, min_y), (max_x, max_y) in [room[0] for room in rooms]:
        for x in range(min_x, max_x + 1):
            if grid[x][min_y] in (18, 1, 0):
                new_grid[x][min_y] = grid[x][min_y]
            if grid[x][max_y] in (18, 1, 0):
                new_grid[x][max_y] = grid[x][max_y]
        for y in range(min_y, max_y + 1):
            if grid[min_x][y] in (18, 1, 0):
                new_grid[min_x][y] = grid[min_x][y]
            if grid[max_x][y] in (18, 1, 0):
                new_grid[max_x][y] = grid[max_x][y]

    for bounds, pixels in rooms:
        (min_x, min_y), (max_x, max_y) = bounds
        freq = defaultdict(int)
        for val in pixels:
            if val not in (18, 1, 255):
                freq[val] += 1
        if not freq:
            continue
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        if sorted_freq:
            dominant = sorted_freq[0][0]
            if dominant == 255 and len(sorted_freq) > 1:
                dominant = sorted_freq[1][0]
        
        for x in range(min_x, max_x + 1): # Fill the room area
            for y in range(min_y, max_y + 1):
                if new_grid[x][y] != 18 and new_grid[x][y] != 1 and new_grid[x][y] != 0:  # Avoid overwriting walls
                    new_grid[x][y] = dominant
    return new_grid

def replace_walls(grid):
    
    Replaces wall pixels (18) between identical regions with the region's value.
    
    SOFT_WALL = 18
    ROOM_WALL = 19
    NOT_ACCESS = {0, 1, 18, 19}
    rows, cols = grid.shape
    new_grid = np.copy(grid)
    WALL_RANGE = 2  # 5x5 area radius
    
    for x in range(rows):
        for y in range(cols):
            left = new_grid[x, y-1] if y > 0 else None
            right = new_grid[x, y+1] if y < cols-1 else None
            rightright = new_grid[x, y+2] if y < cols-2 else None
            rightrightright = new_grid[x, y+3] if y < cols-3 else None
                
            up = new_grid[x-1, y] if x > 0 else None
            down = new_grid[x+1, y] if x < rows-1 else None
            downdown = new_grid[x+2, y] if x < rows-2 else None
            downdowndown = new_grid[x+3, y] if x < rows-3 else None
                
            if new_grid[x, y] == SOFT_WALL:
                # Horizontal checks
                if left == right and left not in NOT_ACCESS:# Replace if horizontal match
                    new_grid[x, y] = left
                elif left == rightright and left not in NOT_ACCESS:
                    new_grid[x, y] = left
                elif right == SOFT_WALL and rightright != SOFT_WALL:
                    new_grid[x, y] = left    
                elif right == SOFT_WALL and rightright == SOFT_WALL and rightrightright != SOFT_WALL:
                    new_grid[x, y] = left
                
                # Vertical checks
                elif up == down and up not in NOT_ACCESS:# Replace if vertical match
                    new_grid[x, y] = up
                elif up == downdown and up not in NOT_ACCESS:
                    new_grid[x, y] = up
                elif down == SOFT_WALL and downdown != SOFT_WALL:
                    new_grid[x, y] = up
                elif down == SOFT_WALL and downdown == SOFT_WALL and downdowndown != SOFT_WALL:
                    new_grid[x, y] = up
                
                # Proximity to outer walls
                elif left == 1:
                    new_grid[x, y] = right
                elif right == 1:
                    new_grid[x, y] = left
                elif up == 1:
                    new_grid[x, y] = down
                elif down == 1:
                    new_grid[x, y] = up

                elif left != right and left not in NOT_ACCESS and right not in NOT_ACCESS:
                    new_grid[x, y] = ROOM_WALL
                elif up != down and up not in NOT_ACCESS and down not in NOT_ACCESS:
                    new_grid[x, y] = ROOM_WALL
            elif new_grid[x, y] == 1:
                if down == ROOM_WALL or up == ROOM_WALL or right == ROOM_WALL or left == ROOM_WALL:
                    new_grid[x, y] = SOFT_WALL 
    for x in range(rows):
        for y in range(cols):
            if new_grid[x, y] == SOFT_WALL:
                # Check in all directions within 5x5 area
                candidates = []
                
                # Horizontal scan
                left_values = [new_grid[x, max(y-i, 0)] for i in range(1, WALL_RANGE+1)]
                right_values = [new_grid[x, min(y+i, cols-1)] for i in range(1, WALL_RANGE+1)]
                
                # Vertical scan
                up_values = [new_grid[max(x-i, 0), y] for i in range(1, WALL_RANGE+1)]
                down_values = [new_grid[min(x+i, rows-1), y] for i in range(1, WALL_RANGE+1)]
                
                # Look for wall-room patterns in 5x5 area
                for distance in [1, 2]:
                    # Check left-wall with right-room pattern
                    if (y-distance >= 0 and new_grid[x, y-distance] == 1 and
                        y+distance < cols and new_grid[x, y+distance] not in NOT_ACCESS):
                        candidates.append(new_grid[x, y+distance])
                    
                    # Check right-wall with left-room pattern
                    if (y+distance < cols and new_grid[x, y+distance] == 1 and
                        y-distance >= 0 and new_grid[x, y-distance] not in NOT_ACCESS):
                        candidates.append(new_grid[x, y-distance])
                    
                    # Check up-wall with down-room pattern
                    if (x-distance >= 0 and new_grid[x-distance, y] == 1 and
                        x+distance < rows and new_grid[x+distance, y] not in NOT_ACCESS):
                        candidates.append(new_grid[x+distance, y])
                    
                    # Check down-wall with up-room pattern
                    if (x+distance < rows and new_grid[x+distance, y] == 1 and
                        x-distance >= 0 and new_grid[x-distance, y] not in NOT_ACCESS):
                        candidates.append(new_grid[x-distance, y])

                # Resolve conflicts: choose most frequent candidate
                if candidates:
                    freq = defaultdict(int)
                    for val in candidates:
                        freq[val] += 1
                    best = max(freq, key=lambda k: (freq[k], k != 255))
                    new_grid[x, y] = best

    return new_grid

def gen_rect_corners(corners):
    Generates rectangles from corner points using neighbor detection.
    x_groups = {}
    y_groups = {}
    corners_set = set(corners)
    
    for point in corners:
        x, y = point
        x_groups.setdefault(y, []).append(x)
        y_groups.setdefault(x, []).append(y)
    
    for y in x_groups:
        x_groups[y].sort()
    for x in y_groups:
        y_groups[x].sort()
    
    rectangles = set()
    
    for point in corners:
        x, y = point
        x_list = x_groups.get(y, [])
        try:
            x_idx = x_list.index(x)
        except ValueError:
            continue
        x_prev = x_list[x_idx-1] if x_idx > 0 else None
        x_next = x_list[x_idx+1] if x_idx < len(x_list)-1 else None
        
        y_list = y_groups.get(x, [])
        try:
            y_idx = y_list.index(y)
        except ValueError:
            continue
        y_prev = y_list[y_idx-1] if y_idx > 0 else None
        y_next = y_list[y_idx+1] if y_idx < len(y_list)-1 else None
        
        neighbor_checks = [
            (x_prev, y_prev),
            (x_prev, y_next),
            (x_next, y_prev),
            (x_next, y_next)
        ]
        
        for nx, ny in neighbor_checks:
            if nx is None or ny is None:
                continue
            if ((nx, ny) in corners_set and 
                (x, ny) in corners_set and 
                (nx, y) in corners_set):
                min_x, max_x = min(x, nx), max(x, nx)
                min_y, max_y = min(y, ny), max(y, ny)
                rectangles.add(((min_x, min_y), (max_x, max_y)))
    new_corners = detect_edge_corners(rectangles, corners_set)
    output = rectangles | gen_rect_corners(new_corners) if new_corners else rectangles
    return output

def detect_edge_corners(rectangles, original_corners):
    Finds new corners created by rectangle edges.
    vertical_edges = set()
    horizontal_edges = set()
    
    for (x1, y1), (x2, y2) in rectangles:
        horizontal_edges.update((x, y1) for x in range(x1, x2 + 1))
        horizontal_edges.update((x, y2) for x in range(x1, x2 + 1))
        vertical_edges.update((x1, y) for y in range(y1, y2 + 1))
        vertical_edges.update((x2, y) for y in range(y1, y2 + 1))
    
    new_corners = vertical_edges & horizontal_edges
    return new_corners - original_corners

def mark_corners_floor(grid):
    Marks corners of the floor map (walls and boundaries).

    corners = []
    rows, cols = grid.shape
    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == 1:  # Wall
                if (x == 0 or x == rows - 1 or y == 0 or y == cols - 1) or (
                    (grid[x - 1][y] != 1 and grid[x][y - 1] != 1) or
                    (grid[x + 1][y] != 1 and grid[x][y - 1] != 1) or
                    (grid[x - 1][y] != 1 and grid[x][y + 1] != 1) or
                    (grid[x + 1][y] != 1 and grid[x][y + 1] != 1)):
                    corners.append((x, y))
    return corners

def mark_corners_T(grid, x_c, y_c, corner_type):

    Detects T-corners from room corners and adds them.

    rows, cols = grid.shape
    t_corners = []
    corner_directions = {
        'top_left': (-1, -1),
        'top_right': (-1, 1),
        'bottom_left': (1, -1),
        'bottom_right': (1, 1)
    }
    if not (0 <= x_c < rows and 0 <= y_c < cols):
        return []
    
    edge_checks = {
        'top_left': (x_c == 0 or y_c == 0),
        'top_right': (x_c == 0 or y_c == cols-1),
        'bottom_left': (x_c == rows-1 or y_c == 0),
        'bottom_right': (x_c == rows-1 or y_c == cols-1)
    }
    if edge_checks.get(corner_type, False):
        return []
    
    dx = corner_directions[corner_type][0]
    current_x = x_c
    while True:
        current_x += dx
        if grid[current_x][y_c] == 18:
            t_corners.append((current_x, y_c))
        if (not (0 <= current_x < rows)) or (grid[current_x][y_c] == 1):
            t_corners.append((current_x, y_c))
            break

    dy = corner_directions[corner_type][1]
    current_y = y_c
    while True:
        current_y += dy
        if grid[x_c][current_y] == 18:
            t_corners.append((x_c, current_y))
        if (not (0 <= current_y < cols)) or (grid[x_c][current_y] == 1):
            t_corners.append((x_c, current_y))
            break

    return t_corners

"""