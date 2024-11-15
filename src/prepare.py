import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(input_path):
    """Load data from a CSV file."""
    df = pd.read_csv(input_path, sep='\s+', header = None)
    # df_labels = pd.read_csv(input_path_labels, sep='s+', header = None)
    return df #, df_labels

def scale_features(df):
    """Standardize dataset."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def save_data(df, output_path):
    """Save the processed DataFrame to a CSV file."""
    df = pd.DataFrame(df)
    df.to_csv(output_path, index=False)

def main(input_path, output_path):
    # Load data
    df = load_data(input_path)
    
    # Scale data
    df = scale_features(df)
    
    # Save processed data
    save_data(df, output_path)
    print(f"Data saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data for analysis")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the processed CSV file")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path)

