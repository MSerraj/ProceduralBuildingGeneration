# ProceduralBuildingGeneration

## Overview

**ProceduralBuildingGeneration** is a Python project for creating and visualizing 2D floorplans using region-growing, stair and corridor placement, and a prototype for Wave Function Collapse (WFC) generation.

### Features
- Convert floorplans into integer-coded grids
- Grow rooms using freeform or Manhattan-style expansions
- Automatically insert windows and stairwells
- Carve corridors with configurable minimum widths
- Procedurally generate patterns using WFC
- Upscale to 3×3 pixel tiles for better visualizations
- Visualize results using Matplotlib with door/seed overlays

## Module Breakdown

<details>
<summary>🧠 Click to expand</summary>

### `utils.py`
- `image_to_int(floorplan) → (np.ndarray, List[Tuple[int, int, int]])`:  
  Converts image to grid and extracts seed points.

- `place_windows(grid, window_value=5, window_size=8, min_spacing=4)`:  
  Automatically places windows along outer walls.

- Plotting: `plot_floorplan` with submodules `_draw_doors`, `_plot_seed_points`, etc.

### `grid.py`
- `Wall (Enum)`: Bitmask values for walls, corners, doors, stairs, etc.
- `convert_to_3x3(grid)`: Logical → 3×3 visual tile upscaling
- `convert_corridor_to_room(grid)`, `convert_3x3_to_3x3int(...)`: Post-processing helpers

### Region-Growing (in `utils.py`)
- `region_growing_simultaneous(grid, seeds)`: 4-connected flood-fill
- `region_growing_simultaneous_rectangular2(grid, seeds)`: Manhattan expansion

### `wfc.py`
- `WFCell`, `WFCGrid`: Simple Wave Function Collapse implementation
- `FloorPlan.generate_wfc(max_attempts)`: Retry until a valid collapse
</details>
