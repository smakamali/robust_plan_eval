import os
import re

def replace_ilike(sql_content):
    """Replace ILIKE with UPPER() equivalents."""
    pattern = re.compile(
        r'(\b[\w\.\(\)]+)\s+ILIKE\s+(\'[^\']*\'|[\w\.\(\)]+)',
        flags=re.IGNORECASE
    )
    return pattern.sub(r'UPPER(\1) LIKE UPPER(\2)', sql_content)

def process_sql_files(root_dir):
    """
    Recursively process all .sql files in directory and its subdirectories,
    replacing ILIKE statements.
    """
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith('.sql'):
                filepath = os.path.join(root, filename)
                try:
                    # Read file content
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Apply transformation
                    modified = replace_ilike(content)
                    
                    # Only write back if changes were made
                    if modified != content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(modified)
                        print(f"Processed: {filepath}")
                    else:
                        print(f"No changes: {filepath}")
                        
                except Exception as e:
                    print(f"Error processing {filepath}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Replace ILIKE with LIKE+UPPER in SQL files"
    )
    parser.add_argument(
        "directory",
        help="Root directory containing SQL files"
    )
    args = parser.parse_args()
    
    if os.path.isdir(args.directory):
        process_sql_files(args.directory)
        print("Processing complete")
    else:
        print(f"Error: {args.directory} is not a valid directory")