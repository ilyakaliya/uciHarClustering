import argparse
import pandas as pd
import yaml
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score, v_measure_score

def load_data(input_path, true_labels_path):
    """Load the dataset and true labels for clustering evaluation."""
    try:
        df = pd.read_csv(input_path)
        true_labels = pd.read_csv(true_labels_path, header=None).values.ravel()  # Load true labels
        print(f"Loaded data with shape: {df.shape}")
        print(f"Loaded true labels with shape: {true_labels.shape}")
    except Exception as e:
        print(f"Error loading data or true labels: {e}")
        raise
    return df, true_labels

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

def evaluate_clustering(df, true_labels, params):
    """Evaluate clustering using the specified metrics."""
    n_clusters = params.get("n_clusters", 3)
    random_state = params.get("random_state", 42)

    print(f"Evaluating KMeans clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(df.iloc[:, :-1])  # Exclude ground truth if present

    metrics = {}
    metrics["silhouette_score"] = silhouette_score(df.iloc[:, :-1], cluster_labels)
    metrics["completeness_score"] = completeness_score(true_labels, cluster_labels)
    metrics["homogeneity_score"] = homogeneity_score(true_labels, cluster_labels)
    metrics["v_measure_score"] = v_measure_score(true_labels, cluster_labels)

    return metrics

def save_metrics(metrics, output_path):
    """Save calculated metrics to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {output_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
        raise

def main(input_path, true_labels_path, params_path, metrics_output_path):
    # Load data and true labels
    df, true_labels = load_data(input_path, true_labels_path)

    # Load parameters
    params = load_params(params_path)

    # Evaluate clustering
    metrics = evaluate_clustering(df, true_labels, params)

    # Save metrics
    save_metrics(metrics, metrics_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate clustering quality")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the reduced dataset CSV file")
    parser.add_argument('--true_labels_path', type=str, required=True, help="Path to the true labels file")
    parser.add_argument('--params_path', type=str, required=True, help="Path to the YAML file with clustering parameters")
    parser.add_argument('--metrics_output_path', type=str, required=True, help="Path to save the evaluation metrics JSON")
    
    args = parser.parse_args()
    main(args.input_path, args.true_labels_path, args.params_path, args.metrics_output_path)
