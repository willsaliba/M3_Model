def read_ldr_file(file_path):
    """
    Reads an LDR file and returns its lines.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def convert_to_relative_coordinates(ldr_lines):
    """
    Converts absolute x, y, z coordinates in LDR lines to relative (delta) coordinates.
    """
    relative_ldr_lines = []
    previous_x, previous_y, previous_z = 0.0, 0.0, 0.0  # Initialize the reference point (origin)

    for line in ldr_lines:
        parts = line.split()

        # Only process lines that define bricks (starting with '1')
        if parts[0] == '1' and len(parts) >= 5:  # Ensure it has enough parts to extract x, y, z
            # Extract absolute x, y, z coordinates (assuming standard LDraw format positions)
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            
            # Compute relative (delta) coordinates
            delta_x = x - previous_x
            delta_y = y - previous_y
            delta_z = z - previous_z
            
            # Update previous coordinates to current brick's coordinates
            previous_x, previous_y, previous_z = x, y, z
            
            # Replace the original absolute coordinates with relative (delta) coordinates
            parts[2], parts[3], parts[4] = f"{delta_x:.2f}", f"{delta_y:.2f}", f"{delta_z:.2f}"
        
        # Append the modified or unmodified line to the result, maintaining newline
        relative_ldr_lines.append(' '.join(parts) + '\n')
    
    return relative_ldr_lines

def convert_to_absolute_coordinates(ldr_lines):
    """
    Converts relative (delta) x, y, z coordinates in LDR lines back to absolute coordinates.
    """
    absolute_ldr_lines = []
    current_x, current_y, current_z = 0.0, 0.0, 0.0  # Initialize starting point at the origin

    for line in ldr_lines:
        parts = line.split()

        # Only process lines that define bricks (starting with '1')
        if parts[0] == '1' and len(parts) >= 5:  # Ensure it has enough parts to extract x, y, z
            # Extract relative (delta) x, y, z coordinates
            delta_x, delta_y, delta_z = float(parts[2]), float(parts[3]), float(parts[4])
            
            # Compute absolute coordinates by adding the deltas to the current coordinates
            x = current_x + delta_x
            y = current_y + delta_y
            z = current_z + delta_z
            
            # Update current coordinates to the newly computed absolute coordinates
            current_x, current_y, current_z = x, y, z
            
            # Replace the relative coordinates with absolute coordinates
            parts[2], parts[3], parts[4] = f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"
        
        # Append the modified or unmodified line to the result, maintaining newline
        absolute_ldr_lines.append(' '.join(parts) + '\n')
    
    return absolute_ldr_lines

def write_ldr_file(file_path, ldr_lines):
    """
    Writes the modified LDR lines to a new file.
    """
    with open(file_path, 'w') as file:
        file.writelines(ldr_lines)

# Main function to process the LDR file
def main(input_file_path, relative_output_file_path, absolute_output_file_path):
    # Read the original LDR file
    ldr_lines = read_ldr_file(input_file_path)
    
    # Convert to relative coordinates
    relative_ldr_lines = convert_to_relative_coordinates(ldr_lines)
    
    # Write the modified LDR lines to a new file with relative coordinates
    write_ldr_file(relative_output_file_path, relative_ldr_lines)
    print(f"Converted LDR file with relative coordinates saved to {relative_output_file_path}")
    
    # Convert back to absolute coordinates
    absolute_ldr_lines = convert_to_absolute_coordinates(relative_ldr_lines)
    
    # Write the modified LDR lines to a new file with absolute coordinates
    write_ldr_file(absolute_output_file_path, absolute_ldr_lines)
    print(f"Converted LDR file back to absolute coordinates saved to {absolute_output_file_path}")

# Example usage:
input_file = 'building17.ldr'  # Replace with your input LDR file path
relative_output_file = 'relative_example.ldr'  # Output file path for relative coordinates
absolute_output_file = 'absolute_example.ldr'  # Output file path for absolute coordinates
main(input_file, relative_output_file, absolute_output_file)
