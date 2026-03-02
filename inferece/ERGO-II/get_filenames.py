import os
import argparse

def get_formatted_filenames(directory, output_file="./filenames.txt"):
    try:
        files = os.listdir(directory)
        csv_files = [os.path.splitext(f)[0] for f in files if f.endswith('.csv')]

        csv_files.sort()

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            for filename in csv_files:
                f.write(f'"{filename}"\n')
                
        print(f"Successfully saved formatted filenames to: {output_file}")
        print(f"Found {len(csv_files)} CSV files")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Format CSV filenames from a directory')
    parser.add_argument('--directory', default="none",help='Directory containing CSV files')
    parser.add_argument('--output', '-o', 
                        default="./filenames.txt",
                        help='Output file path (default: ./filenames.txt)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return

    get_formatted_filenames(args.directory, args.output)

if __name__ == "__main__":
    main()