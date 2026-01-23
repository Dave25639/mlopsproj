"""
Data drift detection for Food-101 dataset.

This module provides functionality to detect data drift between training and test datasets,
or between training data and new incoming data. It uses statistical tests and feature
extraction to identify distribution shifts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats

from mlopsproj.data import Food101DataModule

logger = logging.getLogger(__name__)


def extract_image_statistics(image: Image.Image) -> Dict[str, float]:
    """
    Extract basic statistics from an image.

    Args:
        image: PIL Image object

    Returns:
        Dictionary with image statistics (mean, std, brightness, etc.)
    """
    img_array = np.array(image).astype(np.float32)

    # RGB channel statistics
    r_mean = img_array[:, :, 0].mean()
    g_mean = img_array[:, :, 1].mean()
    b_mean = img_array[:, :, 2].mean()

    r_std = img_array[:, :, 0].std()
    g_std = img_array[:, :, 1].std()
    b_std = img_array[:, :, 2].std()

    # Overall brightness (grayscale conversion)
    gray = 0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean

    # Image size
    width, height = image.size
    aspect_ratio = width / height if height > 0 else 1.0

    return {
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "r_std": r_std,
        "g_std": g_std,
        "b_std": b_std,
        "brightness": gray,
        "aspect_ratio": aspect_ratio,
        "width": float(width),
        "height": float(height),
    }


def extract_features_from_dataset(
    dataset,
    max_samples: Optional[int] = None,
    use_model_features: bool = False,
    model: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    """
    Extract features from a dataset.

    Args:
        dataset: Dataset to extract features from (FoodDataset)
        max_samples: Maximum number of samples to process (None for all)
        use_model_features: If True, use model's feature extractor
        model: Model to use for feature extraction (if use_model_features=True)

    Returns:
        Array of feature vectors (n_samples, n_features)
    """
    features = []
    samples_processed = 0

    # Temporarily disable transforms to get raw images
    original_transform = dataset.transform
    dataset.transform = None

    try:
        for idx in range(min(len(dataset), max_samples if max_samples else len(dataset))):
            if max_samples and samples_processed >= max_samples:
                break

            # Get raw image
            path, label = dataset.samples[idx]
            img = Image.open(path).convert("RGB")

            stats_dict = extract_image_statistics(img)
            features.append(list(stats_dict.values()))
            samples_processed += 1

            if samples_processed % 100 == 0:
                logger.info(f"Processed {samples_processed} samples...")
    finally:
        # Restore original transform
        dataset.transform = original_transform

    return np.array(features)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.

    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate change
    PSI >= 0.25: Significant change

    Args:
        expected: Reference distribution
        actual: Current distribution
        bins: Number of bins for histogram

    Returns:
        PSI value
    """
    # Create bins based on expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Calculate histograms
    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    # Normalize to probabilities
    expected_prob = expected_hist / (expected_hist.sum() + 1e-10)
    actual_prob = actual_hist / (actual_hist.sum() + 1e-10)

    # Calculate PSI
    psi = np.sum((actual_prob - expected_prob) * np.log((actual_prob + 1e-10) / (expected_prob + 1e-10)))

    return float(psi)


def detect_drift(
    reference_features: np.ndarray,
    current_features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """
    Detect drift between reference and current feature distributions.

    Args:
        reference_features: Reference dataset features (n_samples, n_features)
        current_features: Current dataset features (n_samples, n_features)
        feature_names: Names of features (for reporting)
        alpha: Significance level for statistical tests

    Returns:
        Dictionary with drift detection results for each feature
    """
    n_features = reference_features.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    results = {}

    for i, feature_name in enumerate(feature_names):
        ref_feat = reference_features[:, i]
        curr_feat = current_features[:, i]

        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_feat, curr_feat)

        # Population Stability Index
        psi = calculate_psi(ref_feat, curr_feat)

        # Mean difference
        mean_diff = curr_feat.mean() - ref_feat.mean()
        mean_diff_pct = (mean_diff / (ref_feat.mean() + 1e-10)) * 100

        # Standard deviation difference
        std_diff = curr_feat.std() - ref_feat.std()
        std_diff_pct = (std_diff / (ref_feat.std() + 1e-10)) * 100

        # Determine drift status
        has_drift = ks_pvalue < alpha or psi >= 0.25

        results[feature_name] = {
            "ks_statistic": float(ks_statistic),
            "ks_pvalue": float(ks_pvalue),
            "psi": float(psi),
            "mean_diff": float(mean_diff),
            "mean_diff_pct": float(mean_diff_pct),
            "std_diff": float(std_diff),
            "std_diff_pct": float(std_diff_pct),
            "has_drift": has_drift,
            "reference_mean": float(ref_feat.mean()),
            "reference_std": float(ref_feat.std()),
            "current_mean": float(curr_feat.mean()),
            "current_std": float(curr_feat.std()),
        }

    return results


def plot_drift_results(
    reference_features: np.ndarray,
    current_features: np.ndarray,
    drift_results: Dict[str, Dict],
    feature_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot drift detection results.

    Args:
        reference_features: Reference dataset features
        current_features: Current dataset features
        drift_results: Results from detect_drift
        feature_names: Names of features
        output_path: Path to save the plot
    """
    n_features = reference_features.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Create subplots
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        ref_feat = reference_features[:, i]
        curr_feat = current_features[:, i]

        # Plot histograms
        ax.hist(ref_feat, bins=30, alpha=0.5, label="Reference", density=True, color="blue")
        ax.hist(curr_feat, bins=30, alpha=0.5, label="Current", density=True, color="red")

        # Add statistics
        result = drift_results[feature_name]
        title = f"{feature_name}\n"
        title += f"KS p={result['ks_pvalue']:.4f}, PSI={result['psi']:.4f}\n"
        title += f"Drift: {'YES' if result['has_drift'] else 'NO'}"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved drift plot to {output_path}")
    else:
        plt.show()

    plt.close()


def main(
    data_dir: str,
    max_samples: int = 1000,
    output_dir: Optional[str] = None,
) -> None:
    """
    Main function to run drift detection.

    Args:
        data_dir: Path to data directory
        max_samples: Maximum number of samples to analyze
        output_dir: Directory to save results (default: current directory)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(".")

    logger.info("Setting up data module...")
    dm = Food101DataModule(
        data_dir=data_dir,
        batch_size=32,
        num_workers=2,
        val_fraction=0.1,
    )

    logger.info("Loading datasets...")
    dm.setup(stage="fit")
    dm.setup(stage="test")

    logger.info("Extracting features from training data...")
    train_features = extract_features_from_dataset(
        dm.train_dataset,
        max_samples=max_samples
    )

    logger.info("Extracting features from test data...")
    test_features = extract_features_from_dataset(
        dm.test_dataset,
        max_samples=max_samples
    )

    logger.info(f"Training features shape: {train_features.shape}")
    logger.info(f"Test features shape: {test_features.shape}")

    feature_names = [
        "r_mean", "g_mean", "b_mean",
        "r_std", "g_std", "b_std",
        "brightness", "aspect_ratio", "width", "height"
    ]

    logger.info("Detecting drift...")
    drift_results = detect_drift(train_features, test_features, feature_names=feature_names)

    # Print results
    print("\n" + "=" * 80)
    print("DRIFT DETECTION RESULTS")
    print("=" * 80)

    for feature_name, result in drift_results.items():
        print(f"\n{feature_name}:")
        print(f"  KS statistic: {result['ks_statistic']:.4f}")
        print(f"  KS p-value: {result['ks_pvalue']:.4f}")
        print(f"  PSI: {result['psi']:.4f}")
        print(f"  Mean difference: {result['mean_diff']:.4f} ({result['mean_diff_pct']:.2f}%)")
        print(f"  Std difference: {result['std_diff']:.4f} ({result['std_diff_pct']:.2f}%)")
        print(f"  Drift detected: {'YES' if result['has_drift'] else 'NO'}")

    # Count features with drift
    n_drifted = sum(1 for r in drift_results.values() if r["has_drift"])
    print(f"\n{'=' * 80}")
    print(f"Summary: {n_drifted}/{len(drift_results)} features show drift")
    print("=" * 80)

    # Plot results
    plot_path = output_path / "drift_detection.png"
    plot_drift_results(
        train_features,
        test_features,
        drift_results,
        feature_names=feature_names,
        output_path=plot_path
    )

    logger.info("Drift detection complete!")


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    main(data_dir=data_dir, max_samples=max_samples)
