#!/bin/bash

# Create directories for data
mkdir -p raw_data
mkdir -p data

echo "Place your dataset files in the 'raw_data' directory."
echo "The script expects CSV files named according to dataset splits (e.g., train.csv, test.csv, validation.csv)."

# Convert dataset to parquet format
python prepare_dataset.py --input_dir "raw_data" --output_dir "data"

echo "Dataset preparation complete!"

