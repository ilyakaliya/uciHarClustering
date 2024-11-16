import argparse
import pandas as pd
import yaml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def load_data(input_path):
    """Load the scaled dataset."""
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    return df

def save_reduced_data(reduced_data, output_path):
    """Save the reduced dataset."""
    try:
        reduced_df = pd.DataFrame(reduced_data)
        reduced_df.to_csv(output_path, index=False)
        print(f"Reduced data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

def load_params(params_path):
    """Load reduction parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        print(f"Loaded parameters: {params}")
    except Exception as e:
        print(f"Error loading parameters: {e}")
        raise
    return params

def reduce_dimensionality(df, method, params):
    """Apply dimensionality reduction."""
    print(f"Applying {method} dimensionality reduction...")
    if method == "PCA":
        model = PCA(n_components=params.get("n_components", 2))
    elif method == "tSNE":
        model = TSNE(n_components=params.get("n_components", 2),
                     perplexity=params.get("perplexity", 30),
                     learning_rate=params.get("learning_rate", 200),
                     n_iter=params.get("n_iter", 1000),
                     random_state=params.get("random_state", 42))
    elif method == "UMAP":
        model = umap.UMAP(n_components=params.get("n_components", 2),
                          n_neighbors=params.get("n_neighbors", 15),
                          min_dist=params.get("min_dist", 0.1),
                          random_state=params.get("random_state", 42),
                          metric=params.get("metric", "euclidean"))
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    reduced_data = model.fit_transform(df)
    print(f"Dimensionality reduction complete. Reduced shape: {reduced_data.shape}")
    return reduced_data

def main(input_path, output_path, params_path):
    # Load data
    df = load_data(input_path)

    # Load parameters
    params = load_params(params_path)

    # Apply dimensionality reduction
    method = params.get("method", "PCA")
    reduced_data = reduce_dimensionality(df, method, params)

    # Save reduced data
    save_reduced_data(reduced_data, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply dimensionality reduction to a dataset")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input scaled CSV file")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the reduced dataset")
    parser.add_argument('--params_path', type=str, required=True, help="Path to the YAML file with reduction parameters")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.params_path)