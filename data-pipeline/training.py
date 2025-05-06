"""
training.py

Module for benchmarking machine learning models using embeddings from ChromaDB.
Trains and evaluates various models on text embeddings, reporting performance metrics.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logger import logger
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


async def load_embeddings_from_chromadb(
    collection_name: str = "news_articles",
    persist_directory: str = "chroma_db",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Asynchronously load embeddings and labels from ChromaDB.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory where ChromaDB data is stored.

    Returns:
        Tuple containing (embeddings, labels, label_names).
    """
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=persist_directory)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.error("Failed to get collection: %s", e)
        raise

    # Get all items with embeddings
    def fetch_data():
        return collection.get(include=["embeddings", "metadatas"])

    try:
        logger.info("Loading data from ChromaDB collection '%s'...", collection_name)
        data = await asyncio.to_thread(fetch_data)
    except Exception as e:
        logger.error("Failed to fetch data from ChromaDB: %s", e)
        raise

    # Extract embeddings and labels
    embeddings_list = data.get("embeddings", [])
    metadatas_list = data.get("metadatas", [])

    if (
        embeddings_list is None
        or len(embeddings_list) == 0
        or metadatas_list is None
        or len(metadatas_list) == 0
    ):
        logger.error("No embeddings or metadata found in collection")
        raise ValueError("Empty data retrieved from ChromaDB")

    # Extract labels from metadata
    labels = [metadata.get("label", "") for metadata in metadatas_list]

    # Convert to numpy arrays
    embeddings = np.array(embeddings_list)

    # Get unique label names
    unique_labels = sorted(set(str(label) for label in labels))

    # Encode labels as integers
    label_encoder = LabelEncoder()
    encoded_labels = np.array(label_encoder.fit_transform(labels))

    logger.info(
        "Loaded %d samples with %d dimensions", embeddings.shape[0], embeddings.shape[1]
    )
    logger.info("Found %d unique labels: %s", len(unique_labels), unique_labels)

    return embeddings, encoded_labels, unique_labels


def prepare_models(num_classes: int) -> Dict[str, Pipeline]:
    """
    Prepare model pipelines for benchmarking.

    Args:
        num_classes: Number of unique classes in the dataset.

    Returns:
        Dictionary of named model pipelines.
    """
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2", solver="saga", max_iter=1000, random_state=42
                ),
            ),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    probability=True,
                    C=1.0,
                    gamma="scale",
                    random_state=42,
                ),
            ),
        ]),
        "rf": Pipeline([(
            "clf",
            RandomForestClassifier(
                n_estimators=100, max_depth=None, n_jobs=-1, random_state=42
            ),
        )]),
        "kmeans": Pipeline([
            ("scaler", StandardScaler()),
            (
                "cluster",
                KMeans(
                    n_clusters=num_classes, init="k-means++", n_init=10, random_state=42
                ),
            ),
        ]),
        "gmm": Pipeline([
            ("scaler", StandardScaler()),
            (
                "cluster",
                GaussianMixture(
                    n_components=num_classes, covariance_type="full", random_state=42
                ),
            ),
        ]),
    }

    return models


def benchmark_supervised_model(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Benchmark a supervised model, measuring accuracy, F1 score, and timing.

    Args:
        model: Scikit-learn pipeline to evaluate.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        model_name: Name of the model for reporting.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("Training supervised model: %s", model_name)

    # Training time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # Create report
    report = {
        "model": model_name,
        "type": "supervised",
        "accuracy": accuracy,
        "f1_score": f1,
        "train_time": train_time,
        "inference_time": infer_time,
        "primary_metric": f1,  # Using F1 as primary metric
    }

    logger.info("  Accuracy: %.4f, F1 Score: %.4f", accuracy, f1)
    logger.info("  Train time: %.4fs, Inference time: %.4fs", train_time, infer_time)

    return report


def benchmark_unsupervised_model(
    model: Pipeline,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Benchmark an unsupervised model, measuring silhouette score and adjusted Rand index.

    Args:
        model: Scikit-learn pipeline to evaluate.
        x_train: Training features.
        x_test: Test features.
        y_test: Test labels (only used for ARI).
        model_name: Name of the model for reporting.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("Training unsupervised model: %s", model_name)

    # Training time
    start_time = time.time()
    model.fit(x_train)
    train_time = time.time() - start_time

    # Inference time
    start_time = time.time()
    clusters = model.predict(x_test)
    infer_time = time.time() - start_time

    # Calculate metrics
    silhouette = silhouette_score(x_test, clusters)
    ari = adjusted_rand_score(y_test, clusters)

    # Create report
    report = {
        "model": model_name,
        "type": "unsupervised",
        "silhouette_score": silhouette,
        "adjusted_rand_index": ari,
        "train_time": train_time,
        "inference_time": infer_time,
        "primary_metric": ari,  # Using ARI as primary metric
    }

    logger.info("  Silhouette Score: %.4f, ARI: %.4f", silhouette, ari)
    logger.info("  Train time: %.4fs, Inference time: %.4fs", train_time, infer_time)

    return report


async def run_benchmarks(
    selected_models: Optional[List[str]] = None,
    test_size: float = 0.2,
    output_dir: str = "output",
    collection_name: str = "news_articles",
    persist_directory: str = "chroma_db",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run benchmarks on all selected models and return results.

    Args:
        selected_models: List of model names to evaluate (None for all).
        test_size: Proportion of data to use for testing.
        output_dir: Directory to save model artifacts and results.
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory where ChromaDB data is stored.

    Returns:
        Tuple containing (results_df, metadata).
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings and labels
    X, y, label_names = await load_embeddings_from_chromadb(
        collection_name, persist_directory
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info(
        "Split data: %d training samples, %d test samples",
        X_train.shape[0],
        X_test.shape[0],
    )

    # Prepare models
    num_classes = len(set(y))
    all_models = prepare_models(num_classes)

    # Filter models if specified
    if selected_models:
        models = {k: v for k, v in all_models.items() if k in selected_models}
        logger.info("Selected models: %s", list(models.keys()))
    else:
        models = all_models
        logger.info("Using all models: %s", list(models.keys()))

    # Run benchmarks
    results = []

    for name, model in models.items():
        try:
            # Use Python's copy module for deep copying
            import copy

            model_copy = copy.deepcopy(model)

            # Determine if supervised or unsupervised
            if hasattr(model[-1], "predict_proba") or hasattr(model[-1], "classes_"):
                report = benchmark_supervised_model(
                    model_copy, X_train, y_train, X_test, y_test, name
                )
            else:
                report = benchmark_unsupervised_model(
                    model_copy, X_train, X_test, y_test, name
                )

            # Save model
            model_path = run_dir / f"{name}_model.joblib"
            joblib.dump(model_copy, model_path)
            logger.info("Saved model to %s", model_path)

            results.append(report)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error benchmarking model %s: %s", name, e)

    # Create results DataFrame with default columns if empty
    if results:
        results_df = pd.DataFrame(results)
    else:
        # Create empty DataFrame with expected columns
        results_df = pd.DataFrame(
            columns=[
                "model",
                "type",
                "accuracy",
                "f1_score",
                "silhouette_score",
                "adjusted_rand_index",
                "train_time",
                "inference_time",
                "primary_metric",
            ]
        )

    # Save results
    results_path = run_dir / "benchmark_results.csv"
    results_df.to_csv(results_path, index=False)

    # Calculate class distribution
    class_distribution = {}
    for i, label in enumerate(label_names):
        class_distribution[label] = int(np.sum(y == i))

    # Save metadata
    metadata = {
        "run_id": run_id,
        "dataset": {
            "collection_name": collection_name,
            "total_samples": int(len(X)),
            "dimensions": int(X.shape[1]),
            "num_classes": int(num_classes),
            "class_names": label_names,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "class_distribution": class_distribution,
        },
        "models": list(models.keys()),
        "timestamp": datetime.now().isoformat(),
    }

    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved results to %s", run_dir)

    return results_df, metadata


def generate_report(
    results_df: pd.DataFrame,
    output_dir: str,
    dataset_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Generate comprehensive benchmark report with tables and charts.

    Args:
        results_df: DataFrame containing benchmark results.
        output_dir: Directory to save reports.
        dataset_info: Dictionary containing dataset statistics.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Separate supervised and unsupervised models
    supervised_df = results_df[results_df["type"] == "supervised"].copy()
    unsupervised_df = results_df[results_df["type"] == "unsupervised"].copy()

    # Sort by primary metric
    if not supervised_df.empty:
        supervised_df = supervised_df.sort_values("primary_metric", ascending=False)

    if not unsupervised_df.empty:
        unsupervised_df = unsupervised_df.sort_values("primary_metric", ascending=False)

    # Generate dataset statistics visualizations if info is provided
    if dataset_info:
        # Create class distribution pie chart
        if "class_distribution" in dataset_info:
            plt.figure(figsize=(10, 6))
            labels = list(dataset_info["class_distribution"].keys())
            values = list(dataset_info["class_distribution"].values())
            plt.pie(values, labels=labels, autopct="%1.1f%%")
            plt.title("Class Distribution")
            plt.tight_layout()
            plt.savefig(output_path / "class_distribution.png")

        # Create train/test split visualization
        if "train_samples" in dataset_info and "test_samples" in dataset_info:
            plt.figure(figsize=(8, 6))
            split_data = [dataset_info["train_samples"], dataset_info["test_samples"]]
            plt.bar(
                ["Training Set", "Test Set"], split_data, color=["#66b3ff", "#ff9999"]
            )
            for i, v in enumerate(split_data):
                plt.text(i, v + 5, str(v), ha="center")
            plt.title("Dataset Split")
            plt.tight_layout()
            plt.savefig(output_path / "dataset_split.png")

    # Generate model visualizations
    if not results_df.empty:
        plt.figure(figsize=(12, 8))

        # Training time comparison
        plt.subplot(2, 1, 1)
        plt.barh(results_df["model"], results_df["train_time"])
        plt.xlabel("Training Time (s)")
        plt.title("Model Training Time Comparison")
        plt.tight_layout()

        # Inference time comparison
        plt.subplot(2, 1, 2)
        plt.barh(results_df["model"], results_df["inference_time"])
        plt.xlabel("Inference Time (s)")
        plt.title("Model Inference Time Comparison")
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path / "timing_comparison.png")

        # Supervised metrics if available
        if not supervised_df.empty:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(supervised_df))
            width = 0.35

            plt.bar(x - width / 2, supervised_df["accuracy"], width, label="Accuracy")
            plt.bar(x + width / 2, supervised_df["f1_score"], width, label="F1 Score")

            plt.xlabel("Model")
            plt.ylabel("Score")
            plt.title("Supervised Model Performance")
            plt.xticks(x, supervised_df["model"].tolist())
            plt.legend()
            plt.tight_layout()

            plt.savefig(output_path / "supervised_metrics.png")

        # Unsupervised metrics if available
        if not unsupervised_df.empty:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(unsupervised_df))
            width = 0.35

            plt.bar(
                x - width / 2,
                unsupervised_df["silhouette_score"],
                width,
                label="Silhouette",
            )
            plt.bar(
                x + width / 2,
                unsupervised_df["adjusted_rand_index"],
                width,
                label="ARI",
            )

            plt.xlabel("Model")
            plt.ylabel("Score")
            plt.title("Unsupervised Model Performance")
            plt.xticks(x, unsupervised_df["model"].tolist())
            plt.legend()
            plt.tight_layout()

            plt.savefig(output_path / "unsupervised_metrics.png")

    # Create HTML report
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>ML Model Benchmark Results</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        table { border-collapse: collapse; width: 100%; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "        th { background-color: #f2f2f2; }",
        "        tr:nth-child(even) { background-color: #f9f9f9; }",
        "        .section { margin: 30px 0; }",
        "        h2 { color: #333; }",
        "        img { max-width: 100%; height: auto; margin: 20px 0; }",
        "        .stat-container { display: flex; flex-wrap: wrap; gap: 20px; }",
        (
            "        .stat-box { border: 1px solid #ddd; padding: 15px; border-radius:"
            " 5px; flex: 1; min-width: 200px; background-color: #f9f9f9; }"
        ),
        (
            "        .stat-value { font-size: 24px; font-weight: bold; color: #333;"
            " margin: 10px 0; }"
        ),
        "        .stat-label { font-size: 14px; color: #666; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Machine Learning Model Benchmark Results</h1>",
        f"    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]

    # Add dataset statistics section if available
    if dataset_info:
        html_content.extend([
            "    <div class='section'>",
            "        <h2>Dataset Statistics</h2>",
            "        <div class='stat-container'>",
        ])

        # Add general dataset stats
        if "total_samples" in dataset_info:
            html_content.append(f"""
            <div class='stat-box'>
                <div class='stat-label'>Total Samples</div>
                <div class='stat-value'>{dataset_info["total_samples"]}</div>
            </div>
            """)

        if "dimensions" in dataset_info:
            html_content.append(f"""
            <div class='stat-box'>
                <div class='stat-label'>Embedding Dimensions</div>
                <div class='stat-value'>{dataset_info["dimensions"]}</div>
            </div>
            """)

        if "num_classes" in dataset_info:
            html_content.append(f"""
            <div class='stat-box'>
                <div class='stat-label'>Number of Classes</div>
                <div class='stat-value'>{dataset_info["num_classes"]}</div>
            </div>
            """)

        if "train_samples" in dataset_info and "test_samples" in dataset_info:
            train_pct = round(
                100 * dataset_info["train_samples"] / dataset_info["total_samples"]
            )
            test_pct = round(
                100 * dataset_info["test_samples"] / dataset_info["total_samples"]
            )
            html_content.append(f"""
            <div class='stat-box'>
                <div class='stat-label'>Train/Test Split</div>
                <div class='stat-value'>{train_pct}% / {test_pct}%</div>
                <div>Train: {dataset_info["train_samples"]} samples</div>
                <div>Test: {dataset_info["test_samples"]} samples</div>
            </div>
            """)

        html_content.append("        </div>")  # Close stat-container

        # Add class distribution table if available
        if "class_names" in dataset_info and "class_distribution" in dataset_info:
            html_content.extend([
                "        <h3>Class Distribution</h3>",
                "        <table>",
                "            <tr>",
                "                <th>Class</th>",
                "                <th>Count</th>",
                "                <th>Percentage</th>",
                "            </tr>",
            ])

            for class_name, count in dataset_info["class_distribution"].items():
                percentage = round(100 * count / dataset_info["total_samples"], 2)
                html_content.append(f"            <tr>")
                html_content.append(f"                <td>{class_name}</td>")
                html_content.append(f"                <td>{count}</td>")
                html_content.append(f"                <td>{percentage}%</td>")
                html_content.append(f"            </tr>")

            html_content.append("        </table>")

        # Add visualizations
        html_content.extend([
            "        <div style='display: flex; flex-wrap: wrap; gap: 20px;'>",
        ])

        if Path(output_path / "class_distribution.png").exists():
            html_content.append(
                "            <div style='flex: 1; min-width: 300px;'>"
                "<img src='class_distribution.png' alt='Class Distribution'>"
                "</div>"
            )

        if Path(output_path / "dataset_split.png").exists():
            html_content.append(
                "            <div style='flex: 1; min-width: 300px;'>"
                "<img src='dataset_split.png' alt='Dataset Split'>"
                "</div>"
            )

        html_content.append("        </div>")  # Close flex container
        html_content.append("    </div>")  # Close section

    # Add supervised models table
    if not supervised_df.empty:
        html_content.extend([
            "    <div class='section'>",
            "        <h2>Supervised Models</h2>",
            "        <table>",
            "            <tr>",
            "                <th>Model</th>",
            "                <th>Accuracy</th>",
            "                <th>F1 Score</th>",
            "                <th>Training Time (s)</th>",
            "                <th>Inference Time (s)</th>",
            "            </tr>",
        ])

        for _, row in supervised_df.iterrows():
            html_content.append("            <tr>")
            html_content.append(f"                <td>{row['model']}</td>")
            html_content.append(f"                <td>{row['accuracy']:.4f}</td>")
            html_content.append(f"                <td>{row['f1_score']:.4f}</td>")
            html_content.append(f"                <td>{row['train_time']:.4f}</td>")
            html_content.append(f"                <td>{row['inference_time']:.4f}</td>")
            html_content.append("            </tr>")

        html_content.append("        </table>")
        html_content.append(
            "        <img src='supervised_metrics.png' alt='Supervised Model Metrics'>"
        )
        html_content.append("    </div>")

    # Add unsupervised models table
    if not unsupervised_df.empty:
        html_content.extend([
            "    <div class='section'>",
            "        <h2>Unsupervised Models</h2>",
            "        <table>",
            "            <tr>",
            "                <th>Model</th>",
            "                <th>Silhouette Score</th>",
            "                <th>Adjusted Rand Index</th>",
            "                <th>Training Time (s)</th>",
            "                <th>Inference Time (s)</th>",
            "            </tr>",
        ])

        for _, row in unsupervised_df.iterrows():
            html_content.append(f"            <tr>")
            html_content.append(f"                <td>{row['model']}</td>")
            html_content.append(
                f"                <td>{row['silhouette_score']:.4f}</td>"
            )
            html_content.append(
                f"                <td>{row['adjusted_rand_index']:.4f}</td>"
            )
            html_content.append(f"                <td>{row['train_time']:.4f}</td>")
            html_content.append(f"                <td>{row['inference_time']:.4f}</td>")
            html_content.append(f"            </tr>")

        html_content.append("        </table>")
        html_content.append(
            "        <img src='unsupervised_metrics.png' alt='Unsupervised Model"
            " Metrics'>"
        )
        html_content.append("    </div>")

    # Add timing comparison
    html_content.extend([
        "    <div class='section'>",
        "        <h2>Timing Comparison</h2>",
        "        <img src='timing_comparison.png' alt='Model Timing Comparison'>",
        "    </div>",
        "</body>",
        "</html>",
    ])

    # Write HTML report to file
    with open(output_path / "benchmark_report.html", "w") as f:
        f.write("\n".join(html_content))

    logger.info(
        "Generated benchmark report at %s", output_path / "benchmark_report.html"
    )


async def main():
    """Main function to run the benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Benchmark ML models on text embeddings"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logreg", "svm", "rf", "kmeans", "gmm"],
        help="Models to benchmark (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_output",
        help="Output directory for results and models",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="news_articles",
        help="ChromaDB collection name (default: news_articles)",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="chroma_db",
        help="ChromaDB persistence directory (default: chroma_db)",
    )

    args = parser.parse_args()

    try:
        logger.info("Starting benchmarking process")

        # Run benchmarks
        results_df, run_metadata = await run_benchmarks(
            selected_models=args.models,
            test_size=args.test_size,
            output_dir=args.output,
            collection_name=args.collection,
            persist_directory=args.chroma_dir,
        )

        # Generate report
        generate_report(results_df, args.output, run_metadata["dataset"])

        logger.info("Benchmarking completed successfully")

        # Print summary table
        print("\nBenchmark Results Summary:")
        print(
            results_df.sort_values("primary_metric", ascending=False).to_string(
                index=False
            )
        )

    except Exception as e:
        logger.error("Benchmarking failed: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(main())
