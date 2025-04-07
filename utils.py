import numpy as np
from collections import deque, defaultdict
from math import sqrt
import matplotlib.pyplot as plt


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
        (237, 28, 36): 128,      # Seed 1 (red)
        (0, 162, 232): 129,      # Seed 2 (blue)
        (34, 177, 76): 130,      # Seed 3 (green)
        (163, 73, 164): 132,    # Seed 4 (purple)
        (255, 127, 39): 136,    # Seed 5 (orange)
        (255, 242, 0): 144     # Seed 6 (yellow)
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

def plot_floorplan(output_array, seed_coordinates=None, save=False):
    """
    Plots the processed floorplan and highlights seed coordinates.

    Args:
        output_array (np.array): Processed floorplan array.
        seed_coordinates (list): List of seed coordinates in the format [(x1, y1, value1), ...].
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(output_array, cmap="viridis", vmin=0, vmax=255)
    plt.colorbar(label="Pixel Value")
    plt.title("Processed Floorplan")
    if (seed_coordinates):
        for seed in seed_coordinates:
            print(seed)
            x, y, value = seed
            plt.scatter(x, y, color="red", s=50, edgecolors="white", label=f"Seed {value}")
            plt.text(x, y, f"{value}", color="white", fontsize=12, ha="center", va="center")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.axis('off')
    if (save):
        filename = f"Floor_{seed_coordinates[0][0]}.png"  # Use the first seed's x-coordinate in the filename
        plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save with high resolution and tight bounding box
    plt.show()

    plt.close()  # Close the figure to free up memory



from collections import deque
import numpy as np

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

def fit_largest_rectangle(grid, room_number):
    """ 
    Fit largest rectangle in room with room cells.
    Returns the coordinates of the largest rectangle as (top, left, bottom, right).
    """
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
    """
    Builds a straight wall between points a and b on the grid.
    Walls can be either horizontal or vertical.
    """
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
    """
    For each room number, find its largest rectangle and build walls around it.
    """
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

    grid = fill_rooms_with_dominant_color(grid)
    grid = replace_walls(grid)
    grid = replace_walls(grid)
    grid = replace_walls(grid)
    return grid

def find_rooms(grid):
    """Identify rooms using flood fill, separated by walls (18)."""
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
    """Fill each room with its dominant color (excluding walls)."""
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
    """
    Replaces wall pixels (18) between identical regions with the region's value.
    """
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
    """Generates rectangles from corner points using neighbor detection."""
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
    """Finds new corners created by rectangle edges."""
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
    """
    Marks corners of the floor map (walls and boundaries).
    """
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
    """
    Detects T-corners from room corners and adds them.
    """
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

def int_to_color(result):
    """
    Converts integer grid values to RGB colors.
    """
    reverse_mapping = {
        18: (40, 81, 81),       # Wall color
        19: (180, 180, 180),    # Room separator
        20: (128, 64, 64),      # Escalator/stairs
        21: (128, 0, 64),       # Corridor
        128: (237, 28, 36),     # Seed 1 (red)
        129: (0, 162, 232),     # Seed 2 (blue)
        130: (34, 177, 76),     # Seed 3 (green)
        132: (163, 73, 164),    # Seed 4 (purple)
        136: (255, 127, 39),    # Seed 5 (orange)
        144: (255, 242, 0)      # Seed 6 (yellow)
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


################# STAIRWELL ####################

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
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1), 
                   (1, -1),  (1, 0),  (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
            if skeleton[ny, nx]:
                neighbors.append((ny, nx))
    return neighbors

def bfs_on_skeleton(skeleton, start_nodes, target_set):
    """
    Perform a multi-source BFS from start_nodes over skeleton pixels until
    any pixel in target_set is reached. Returns the shortest path (as a list
    of (y,x) coordinates) from one of the start nodes to the target.
    """
    queue = deque()
    visited = set()
    parent = {}  # For path reconstruction
    
    for node in start_nodes:
        queue.append(node)
        visited.add(node)
        parent[node] = None
    
    while queue:
        current = queue.popleft()
        if current in target_set:
            # Reconstruct path from the start to current.
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        for nb in get_neighbors(current, skeleton):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = current
                queue.append(nb)
    return None  # No connection found

def find_room_boundaries(grid, room_value, skeleton):
    """
    Returns a list of skeleton pixel coordinates that are adjacent
    (4-connected) to any cell belonging to the room (or stairwell).
    """
    bounds = set()
    room_cells = np.argwhere(grid == room_value)
    for y, x in room_cells:
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1), 
                       (1, -1),  (1, 0),  (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                if skeleton[ny, nx]:
                    bounds.add((ny, nx))
    return list(bounds)

def find_optimal_corridor_tree(grid):
    """
    Constructs a minimal corridor tree (corridor pixels marked as 21)
    that connects the stairwell (20) to all rooms (all values except 0,1,18,19,20,21).
    The algorithm uses the skeleton of possible corridor locations and a
    multi-target BFS to add the shortest connection from the growing tree
    to each unconnected room.
    """
    # Build the skeleton (corridors allowed where grid==1, 18, or 19)
    skeleton = np.where((grid == 1) | (grid == 18) | (grid == 19), 1, 0).astype(np.uint8)
    
    # Identify room values (exclude corridors, special values, and stairwell)
    rooms = [val for val in np.unique(grid) if val not in {0, 1, 18, 19, 21} and val != 20]
    
    stair_pos = np.argwhere(grid == 20)
    if len(stair_pos) == 0:
        raise ValueError("No stairwell (20) found in grid")
    
    boundaries = {}
    for room in rooms + [20]:
        boundaries[room] = find_room_boundaries(grid, room, skeleton)
    print(boundaries)
    
    # --- Initialization ---
    # Start with a single seed from the stairwell boundary.
    if not boundaries[20]:
        raise ValueError("Stairwell has no valid boundary on the skeleton")
    
    tree_nodes = set([boundaries[20][0]])  # our initial tree (a single pixel)
    connected_rooms = {20}  # the stairwell is our seed
    unconnected_rooms = set(rooms)  # remaining rooms
    
    # Build mapping for each room: room id -> set of boundary pixels (as candidates)
    target_boundaries = {room: set(boundaries[room]) for room in unconnected_rooms}
    
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
        grid[y, x] = 21


    return widen_corridors(grid)

def widen_corridors(grid, val = 21, min_width = 4):
    """ 
    Expands corridor to minimum width
    """
    corridor_mask = (grid==val)
    valid_mask = (grid != 0)

    iterations = (min_width) // 2

    kernel = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]
    
    for _ in range(iterations):
        expansion = np.zeros_like(corridor_mask)
        for dy, dx in kernel:
            shifted = np.roll(corridor_mask, shift = (dy, dx), axis = (0,1 ))
            expansion |= shifted

        expansion &= valid_mask
        expansion &= ~corridor_mask
        corridor_mask |= expansion

    grid[corridor_mask] = val
    return grid
