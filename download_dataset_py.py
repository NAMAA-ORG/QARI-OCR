import os
import pandas as pd
import argparse
import glob

def prepare_dataset_directory(data_dir="data"):
    """
    Prepare the dataset directory structure for parquet files.
    
    Args:
        data_dir (str): Directory to create for dataset files
    """
    # Create output directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created directory: {data_dir}")

def convert_to_parquet(input_dir, output_dir="data"):
    """
    Convert dataset files to parquet format.
    
    Args:
        input_dir (str): Directory containing the input files
        output_dir (str): Directory to save the parquet files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all relevant files in the input directory
    for file_path in glob.glob(os.path.join(input_dir, "*.csv")):
        file_name = os.path.basename(file_path)
        split_name = os.path.splitext(file_name)[0]  # Remove extension
        output_file = os.path.join(output_dir, f"{split_name}.parquet")
        
        # Read and convert the file
        print(f"Converting {file_path} to parquet...")
        df = pd.read_csv(file_path)
        df.to_parquet(output_file, index=False)
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset directories and convert files to parquet format")
    parser.add_argument("--input_dir", type=str, default="raw_data", help="Directory containing raw dataset files")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for processed dataset files")
    
    args = parser.parse_args()
    
    # Prepare dataset directory
    prepare_dataset_directory(args.output_dir)
    
    # Convert files if input directory exists
    if os.path.exists(args.input_dir):
        convert_to_parquet(args.input_dir, args.output_dir)
    else:
        print(f"Input directory '{args.input_dir}' not found. Please create it and add your dataset files.")

