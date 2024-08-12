import os
import numpy as np
from pathlib import Path
"""
This one sorts by first the bricks height -y, then disrance from 0 (+x, + z)
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
    orientation = parts[5:14]  # The 3x3 rotation matrix parts
    return color, shape, x, y, z, orientation

def sort_bricks_in_file(lines):
    """
    Sorts only the LEGO brick lines in an LDraw file and keeps other lines intact.
    
    Args:
    - lines (list of str): The lines from an LDraw file.
    
    Returns:
    - list of str: The lines with sorted brick lines based on the defined criteria.
    """
    # Separate brick lines from other lines
    brick_lines = []
    other_lines = []
    
    for line in lines:
        if parse_brick_line(line):
            brick_lines.append(line)
        else:
            other_lines.append(line)
    
    # Sort brick lines based on height (y), then distance from center (sqrt(x^2 + z^2))
    brick_data = [parse_brick_line(line) for line in brick_lines]
    brick_data.sort(key=lambda data: (-data[3], np.sqrt(data[2]**2 + data[4]**2)))

    # Reconstruct the sorted brick lines
    sorted_brick_lines = []
    for data in brick_data:
        color, shape, x, y, z, orientation = data
        line_parts = ['1', color, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"] + orientation + [shape]
        sorted_line = ' '.join(line_parts)
        sorted_brick_lines.append(sorted_line)
    
    # Combine the other lines with the sorted brick lines and ensure each line ends with a newline character
    result_lines = other_lines + [line + '\n' for line in sorted_brick_lines]
    return result_lines

def process_directory(source_dir, target_dir):
    """
    Processes all LDraw files in the source directory, sorts the brick lines, and saves them in the target directory.
    
    Args:
    - source_dir (str or Path): The directory containing the original LDraw files.
    - target_dir (str or Path): The directory where the sorted LDraw files should be saved.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True) 
    
    # Process each file in the source directory
    for file_path in source_dir.iterdir():
        if file_path.suffix.lower() in ['.mpd', '.ldr']:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            sorted_lines = sort_bricks_in_file(lines)
            
            # Write the sorted content to the new file in the target directory
            sorted_file_path = target_dir / file_path.name
            with open(sorted_file_path, 'w', encoding='utf-8') as file:
                file.writelines(sorted_lines)
            print(f"Processed and saved sorted file: {sorted_file_path}")

source_directory = 'datasets/omr_whole_slide_50_5step/test' 
target_directory = 'datasets/sorted_omr_whole_slide_50_5step/test'   

process_directory(source_directory, target_directory)
