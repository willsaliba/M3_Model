import os
import glob
import math

def read_ldr_file(file_path):
    """
    Reads an LDR file and returns its lines.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_brick(line):
    """
    Parses an LDR line to extract brick information (color, x, y, z, a, b, c, d, e, f, g, h, i, file).
    """
    parts = line.split()
    if len(parts) >= 15 and parts[0] == '1':  # Ensure it has enough parts and starts with '1'
        color = parts[1]
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        matrix = parts[5:14]  # Transformation matrix
        file = parts[14]
        return (color, x, y, z, matrix, file)
    return None

def calculate_distance(x1, y1, z1, x2=0, y2=0, z2=0):
    """
    Calculates the Euclidean distance between two points (x1, y1, z1) and (x2, y2, z2).
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def sort_bricks_by_closest(ldr_lines):
    """
    Sorts the LDR bricks by the closest distance starting from the origin.
    """
    bricks = []
    non_brick_lines = []

    # Parse LDR lines into bricks and non-brick lines
    for line in ldr_lines:
        parsed_brick = parse_brick(line)
        if parsed_brick:
            bricks.append((parsed_brick, line))  # Store both the brick and the original line
        else:
            non_brick_lines.append(line)

    # Start sorting by closest brick to the origin
    sorted_bricks = []
    if bricks:
        # Initialize with the closest brick to the origin
        first_brick = min(bricks, key=lambda b: calculate_distance(b[0][1], b[0][2], b[0][3]))
        sorted_bricks.append(first_brick)
        bricks.remove(first_brick)

        # Iteratively find the closest brick to the last added brick
        while bricks:
            last_brick = sorted_bricks[-1][0]  # Get the last brick's data
            next_brick = min(bricks, key=lambda b: calculate_distance(b[0][1], b[0][2], b[0][3], last_brick[1], last_brick[2], last_brick[3]))
            sorted_bricks.append(next_brick)
            bricks.remove(next_brick)

    # Combine sorted bricks and non-brick lines
    sorted_ldr_lines = non_brick_lines + [brick[1] for brick in sorted_bricks]
    return sorted_ldr_lines

def write_ldr_file(file_path, ldr_lines):
    """
    Writes the modified LDR lines to a new file.
    """
    with open(file_path, 'w') as file:
        file.writelines(ldr_lines)
def process_directory(input_directory, sorted_output_directory):
    """
    Processes all LDR files in the input directory, sorts the bricks by the closest distance,
    and writes the output to the sorted output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(sorted_output_directory, exist_ok=True)
    
    # Find all .ldr files in the input directory
    ldr_files = glob.glob(os.path.join(input_directory, '*.ldr'))
    
    for ldr_file in ldr_files:
        # Read the original LDR file
        ldr_lines = read_ldr_file(ldr_file)
        
        # Sort bricks by the closest distance
        sorted_ldr_lines = sort_bricks_by_closest(ldr_lines)
        
        # Prepare output file path
        base_filename = os.path.basename(ldr_file)
        sorted_output_file_path = os.path.join(sorted_output_directory, f"sorted_{base_filename}")
        
        # Write the sorted LDR lines to a new file
        write_ldr_file(sorted_output_file_path, sorted_ldr_lines)
        print(f"Sorted LDR file saved to {sorted_output_file_path}")

# Example usage:
input_directory = 'ldr_files'  # Replace with your input directory containing LDR files
sorted_output_directory = 'sorted_output'  # Directory to save sorted LDR files
process_directory(input_directory, sorted_output_directory)

# Main function to process the LDR file
def main(input_file_path, sorted_output_file_path):
    # Read the original LDR file
    ldr_lines = read_ldr_file(input_file_path)
    
    # Sort bricks by the closest distance
    sorted_ldr_lines = sort_bricks_by_closest(ldr_lines)
    
    # Write the sorted LDR lines to a new file
    write_ldr_file(sorted_output_file_path, sorted_ldr_lines)
    print(f"Sorted LDR file saved to {sorted_output_file_path}")

# Example usage:
input_directory = 'data/test'  # Replace with your input directory containing LDR files
sorted_output_directory = 'sorted_output'  # Directory to save sorted LDR files
process_directory(input_directory, sorted_output_directory)
