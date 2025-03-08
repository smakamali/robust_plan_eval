# write a code to iterate over files in a directory `dir_name` and rename each file with a name in the formar `dir_name`_`i`.pickle where `i` is the index of the file in the directory.

import os

def rename_files_in_directory(dir_name):
    files = sorted(os.listdir(dir_name))
    base_dir = os.path.basename(dir_name)  # store the basename separately
    for i, filename in enumerate(files):
        old_file_path = os.path.join(dir_name, filename)
        new_file_name = f"{base_dir}_{i}.sql"  # update extension if needed (e.g., .pickle)
        new_file_path = os.path.join(dir_name, new_file_name)
        os.rename(old_file_path, new_file_path)

def rename_files_in_subdirectories(dir_name):
    for sub_dir in os.listdir(dir_name):
        sub_dir_path = os.path.join(dir_name, sub_dir)
        if os.path.isdir(sub_dir_path):
            rename_files_in_directory(sub_dir_path)

# Example usage
rename_files_in_subdirectories('./input/ceb-imdb-13k')

