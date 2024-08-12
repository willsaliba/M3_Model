import os
import numpy as np
from pathlib import Path

"""
This sort finds the closest brick to the origin, and then sorts the rest based on their distance from this
first brick
"""
def parse_brick_line(line):
    """
    Parses a line from an LDraw file and returns the position, orientation, and other metadata if it's a brick line.
    
    Args:
    - line (str): A string from an LDraw file.
    
    Returns:
    - tuple: A tuple containing color, shape, position (x, y, z), and orientation if it's a brick line, else None.
    """
    parts = line.strip().split()
    if len(parts) < 15 or parts[0] != '1':
        return None 
    
    color = parts[1]
    shape = parts[-1]
    x, y, z = map(float, parts[2:5])
    orientation = parts[5:14] 
    return color, shape, x, y, z, orientation

def format_coordinate(value):
    """
    Formats the coordinate by removing unnecessary .000 if the value is an integer.
    
    Args:
    - value (float): The coordinate value.
    
    Returns:
    - str: The formatted coordinate as a string.
    """
    if value.is_integer():
        return f"{int(value)}"
    else:
        return f"{value:.6f}".rstrip('0').rstrip('.')

def sort_bricks_by_proximity(bricks):
    """
    Sort bricks by proximity, first to the origin, then to the previous brick.
    
    Args:
    - bricks (list of tuples): Parsed brick data (color, shape, x, y, z, orientation).
    
    Returns:
    - list of tuples: Sorted list of brick data.
    """
    # Start by finding the brick closest to the origin (0, 0, 0)
    sorted_bricks = []
    current_brick = min(bricks, key=lambda b: np.sqrt(b[2]**2 + b[3]**2 + b[4]**2))
    sorted_bricks.append(current_brick)
    bricks.remove(current_brick)
    
    # Sort the rest based on their proximity to the previous brick
    while bricks:
        next_brick = min(bricks, key=lambda b: np.sqrt((b[2] - current_brick[2])**2 +
                                                       (b[3] - current_brick[3])**2 +
                                                       (b[4] - current_brick[4])**2))
        sorted_bricks.append(next_brick)
        bricks.remove(next_brick)
        current_brick = next_brick
    
    return sorted_bricks

def sort_bricks_in_file(lines):
    """
    Sorts only the LEGO brick lines in an LDraw file based on their proximity, 
    first to the origin, then to the previous brick.
    
    Args:
    - lines (list of str): The lines from an LDraw file.
    
    Returns:
    - list of str: The lines with sorted brick lines based on proximity.
    """
    # Separate brick lines from other lines
    brick_lines = []
    other_lines = []
    
    for line in lines:
        if parse_brick_line(line):
            brick_lines.append(line)
        else:
            other_lines.append(line)
    
    # Parse brick lines and sort by proximity
    brick_data = [parse_brick_line(line) for line in brick_lines]
    sorted_brick_data = sort_bricks_by_proximity(brick_data)

    # Reconstruct the sorted brick lines
    sorted_brick_lines = []
    for data in sorted_brick_data:
        color, shape, x, y, z, orientation = data
        line_parts = [
            '1', 
            color, 
            format_coordinate(x), 
            format_coordinate(y), 
            format_coordinate(z)
        ] + orientation + [shape]
        sorted_line = ' '.join(line_parts)
        sorted_brick_lines.append(sorted_line)
    
    # Combine the other lines with the sorted brick lines and ensure each line ends with a newline character
    result_lines = other_lines + [line + '\n' for line in sorted_brick_lines]
    return result_lines

def process_directory(source_dir, target_dir):
    """
    Processes all LDraw files in the source directory, sorts the brick lines by proximity,
    and saves them in the target directory.
    
    Args:
    - source_dir (str or Path): The directory containing the original LDraw files.
    - target_dir (str or Path): The directory where the sorted LDraw files should be saved.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)  # Ensure the target directory exists
    
    # Process each file in the source directory
    for file_path in source_dir.iterdir():
        if file_path.suffix.lower() in ['.mpd', '.ldr']:
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Sort the brick lines
            sorted_lines = sort_bricks_in_file(lines)
            
            # Write the sorted content to the new file in the target directory
            sorted_file_path = target_dir / file_path.name
            with open(sorted_file_path, 'w', encoding='utf-8') as file:
                file.writelines(sorted_lines)
            print(f"Processed and saved sorted file: {sorted_file_path}")

# Example usage
source_directory = ''  # Replace with your source directory path
target_directory = ''   # Replace with your target directory path

process_directory(source_directory, target_directory)
