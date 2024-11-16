import argparse
import pandas as pd
import yaml
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(input_path):
    """Load the reduced dataset."""
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    return df

def save_clustered_data(df, output_path):
    """Save the clustered dataset."""
    try:
        df.to_csv(output_path, index=False)
        print(f"Clustered data saved to {output_path}")
    except Exception as e:
        print(f"Error saving clustered data: {e}")
        raise

def load_params(params_path):
    """Load clustering parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        print(f"Loaded parameters: {params}")
    except Exception as e:
        print(f"Error loading parameters: {e}")
        raise
    return params

def apply_kmeans(df, params):
    """Apply KMeans clustering."""
    n_clusters = params.get("n_clusters", 3)
    random_state = params.get("random_state", 42)

    print(f"Applying KMeans clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(df)

    print("Clustering completed.")
    return clusters, kmeans

def visualize_clusters_with_labels(df, clusters, true_labels, output_path):
    """
    Visualize clusters in a 2D scatter plot and compare them with true labels.
    :param df: DataFrame containing the reduced dataset.
    :param clusters: Array of cluster labels predicted by KMeans.
    :param true_labels: Array of true labels.
    :param output_path: Path to save the visualization.
    """
    try:
        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot the true labels
        sns.scatterplot(
            ax=axes[0],
            x=df.iloc[:, 0],
            y=df.iloc[:, 1],
            hue=true_labels.squeeze(),
            palette="tab10",
            s=50,
            alpha=0.8
        )
        axes[0].set_title("True Labels")
        axes[0].set_xlabel("Component 1")
        axes[0].set_ylabel("Component 2")
        axes[0].legend(title="Label", loc='upper right', bbox_to_anchor=(1.2, 1))

        # Plot the predicted clusters
        sns.scatterplot(
            ax=axes[1],
            x=df.iloc[:, 0],
            y=df.iloc[:, 1],
            hue=clusters,
            palette="viridis",
            s=50,
            alpha=0.8
        )
        axes[1].set_title("Predicted Clusters")
        axes[1].set_xlabel("Component 1")
        axes[1].set_ylabel("Component 2")
        axes[1].legend(title="Cluster", loc='upper right', bbox_to_anchor=(1.2, 1))
        
        # Adjust layout and save the visualization
        plt.tight_layout()
        plot_path = output_path.replace(".csv", "_clusters_vs_labels.png")
        plt.savefig(plot_path)
        print(f"Cluster vs. true labels visualization saved to {plot_path}")
    except Exception as e:
        print(f"Error visualizing clusters and labels: {e}")
        raise

def main(input_path, output_path, params_path, true_labels_path):
    # Load data
    df = load_data(input_path)

    # Load parameters
    params = load_params(params_path)

    # Load true labels
    try:
        true_labels = pd.read_csv(true_labels_path, header=None).values
        print(f"Loaded true labels with shape: {true_labels.shape}")
    except Exception as e:
        print(f"Error loading true labels: {e}")
        raise

    # Apply KMeans clustering
    clusters, kmeans = apply_kmeans(df, params)

    # Add cluster labels to the DataFrame
    df["Cluster"] = clusters

    # Save clustered data
    save_clustered_data(df, output_path)

    # Visualize clusters and compare with true labels
    visualize_clusters_with_labels(df.iloc[:, :-1], clusters, true_labels, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply KMeans clustering to a dataset")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the reduced dataset CSV file")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the clustered dataset")
    parser.add_argument('--params_path', type=str, required=True, help="Path to the YAML file with clustering parameters")
    parser.add_argument('--true_labels_path', type=str, required=True, help="Path to the true labels file")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.params_path, args.true_labels_path)
