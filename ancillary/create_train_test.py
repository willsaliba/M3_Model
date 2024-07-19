# import os

# def rename_files_in_directory(directory):
#     # Get a list of all files in the directory
#     files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
#     # Iterate over the files and rename them
#     for index, filename in enumerate(files):
#         # Get the file extension
#         file_extension = os.path.splitext(filename)[1]
        
#         # Create the new file name
#         new_name = f"vehicle{index + 1}{file_extension}"
        
#         # Get the full file paths
#         old_file_path = os.path.join(directory, filename)
#         new_file_path = os.path.join(directory, new_name)
        
#         # Rename the file
#         os.rename(old_file_path, new_file_path)
        
#         print(f"Renamed '{filename}' to '{new_name}'")

# # Specify the directory
# directory_path = 'split_data/vehicles'

# # Call the function to rename files
# rename_files_in_directory(directory_path)

# import os
# import shutil
# import random
# import math

# def split_files(directory_list, train_dir='train', test_dir='test'):
#     for directory in directory_list:
#         # Ensure directory exists
#         if not os.path.isdir(directory):
#             print(f"Directory {directory} does not exist.")
#             continue

#         # List all files in the directory
#         all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
#         # Shuffle the list of files for random selection
#         random.shuffle(all_files)

#         # Calculate the number of files for training and testing
#         train_count = math.ceil(len(all_files) * 0.95)
#         test_count = len(all_files) - train_count

#         # Split the files into train and test sets
#         train_files = all_files[:train_count]
#         test_files = all_files[train_count:train_count + test_count]

#         # Create train and test directories if they don't exist
#         os.makedirs(train_dir, exist_ok=True)
#         os.makedirs(test_dir, exist_ok=True)

#         # Move the train files
#         for file in train_files:
#             shutil.move(os.path.join(directory, file), os.path.join(train_dir, file))

#         # Move the test files
#         for file in test_files:
#             shutil.move(os.path.join(directory, file), os.path.join(test_dir, file))

#         print(f"Processed directory {directory}: {train_count} files moved to {train_dir}, {test_count} files moved to {test_dir}")

# # List of directories to process
# directories_to_process = ['split_data/buildings', 'split_data/creatures', 'split_data/nature', 'split_data/vehicles']

# # Call the function with the list of directories
# split_files(directories_to_process)
