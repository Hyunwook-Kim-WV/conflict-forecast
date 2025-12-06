"""
Evaluation metrics and conflict detection
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("evaluator")


class ConflictEvaluator:
    """Evaluate conflict prediction performance"""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize evaluator

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Evaluator initialized. Output: {self.output_dir}")

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics

        Args:
            y_true: True labels (0=normal, 1=conflict)
            y_pred: Predicted labels
            y_scores: Prediction scores (optional, for AUC)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except ValueError:
                metrics['roc_auc'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negative'] = tn
            metrics['false_positive'] = fp
            metrics['false_negative'] = fn
            metrics['true_positive'] = tp

            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        logger.info("Evaluation Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str,
        title: str = "Confusion Matrix"
    ):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Conflict'],
            yticklabels=['Normal', 'Conflict']
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: str,
        title: str = "ROC Curve"
    ):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved ROC curve to {save_path}")

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: str,
        title: str = "Precision-Recall Curve"
    ):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved PR curve to {save_path}")

    def plot_anomaly_timeline(
        self,
        dates: pd.Series,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        threshold: float,
        save_path: str,
        title: str = "Conflict Detection Timeline"
    ):
        """
        Plot anomaly detection timeline

        Args:
            dates: Date series
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Anomaly scores
            threshold: Detection threshold
            save_path: Path to save plot
            title: Plot title
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Plot 1: Anomaly scores
        axes[0].plot(dates, y_scores, label='Anomaly Score', alpha=0.7)
        axes[0].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        axes[0].set_ylabel('Reconstruction Error')
        axes[0].set_title('Anomaly Scores Over Time')
        axes[0].legend()
        axes[0].grid(True)

        # Plot 2: True labels
        conflict_dates_true = dates[y_true == 1]
        axes[1].scatter(conflict_dates_true, [1]*len(conflict_dates_true),
                       marker='|', s=100, c='red', label='Conflict Period')
        axes[1].set_ylabel('True Label')
        axes[1].set_ylim(-0.5, 1.5)
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['Normal', 'Conflict'])
        axes[1].set_title('Ground Truth')
        axes[1].grid(True)

        # Plot 3: Predictions
        conflict_dates_pred = dates[y_pred == 1]
        axes[2].scatter(conflict_dates_pred, [1]*len(conflict_dates_pred),
                       marker='|', s=100, c='orange', label='Detected Anomaly')
        axes[2].set_ylabel('Prediction')
        axes[2].set_xlabel('Date')
        axes[2].set_ylim(-0.5, 1.5)
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(['Normal', 'Conflict'])
        axes[2].set_title('Model Predictions')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved timeline plot to {save_path}")

    def plot_error_distribution(
        self,
        normal_errors: np.ndarray,
        conflict_errors: np.ndarray,
        threshold: float,
        save_path: str,
        title: str = "Reconstruction Error Distribution"
    ):
        """Plot error distribution for normal vs conflict"""
        plt.figure(figsize=(10, 6))

        plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal', color='blue')
        plt.hist(conflict_errors, bins=50, alpha=0.6, label='Conflict', color='red')
        plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')

        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved error distribution to {save_path}")

    def generate_report(
        self,
        region_name: str,
        metrics: Dict,
        dates: pd.Series,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        threshold: float
    ):
        """
        Generate comprehensive evaluation report

        Args:
            region_name: Name of region
            metrics: Evaluation metrics
            dates: Dates
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Anomaly scores
            threshold: Detection threshold
        """
        region_dir = self.output_dir / region_name
        region_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics to file
        metrics_file = region_dir / 'metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write(f"Evaluation Report: {region_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Threshold: {threshold:.6f}\n\n")
            f.write("Metrics:\n")
            for key, value in metrics.items():
                if key != 'confusion_matrix':
                    f.write(f"  {key}: {value}\n")

            f.write("\nClassification Report:\n")
            f.write(classification_report(y_true, y_pred,
                                         target_names=['Normal', 'Conflict']))

        logger.info(f"Saved metrics to {metrics_file}")

        # Plot confusion matrix
        if 'confusion_matrix' in metrics:
            cm_path = region_dir / 'confusion_matrix.png'
            self.plot_confusion_matrix(
                metrics['confusion_matrix'],
                str(cm_path),
                title=f"Confusion Matrix - {region_name}"
            )

        # Plot ROC curve
        if len(np.unique(y_true)) > 1:
            roc_path = region_dir / 'roc_curve.png'
            self.plot_roc_curve(
                y_true, y_scores,
                str(roc_path),
                title=f"ROC Curve - {region_name}"
            )

            # Plot PR curve
            pr_path = region_dir / 'pr_curve.png'
            self.plot_precision_recall_curve(
                y_true, y_scores,
                str(pr_path),
                title=f"Precision-Recall Curve - {region_name}"
            )

        # Plot timeline
        timeline_path = region_dir / 'timeline.png'
        self.plot_anomaly_timeline(
            dates, y_true, y_pred, y_scores, threshold,
            str(timeline_path),
            title=f"Conflict Detection Timeline - {region_name}"
        )

        # Plot error distribution
        normal_errors = y_scores[y_true == 0]
        conflict_errors = y_scores[y_true == 1]
        if len(normal_errors) > 0 and len(conflict_errors) > 0:
            error_dist_path = region_dir / 'error_distribution.png'
            self.plot_error_distribution(
                normal_errors, conflict_errors, threshold,
                str(error_dist_path),
                title=f"Error Distribution - {region_name}"
            )

        logger.info(f"Generated complete report for {region_name}")


if __name__ == "__main__":
    # Example usage
    evaluator = ConflictEvaluator()

    # Simulate data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    y_true = np.random.binomial(1, 0.2, n_samples)
    y_scores = np.random.random(n_samples)
    y_scores[y_true == 1] += 0.3  # Higher scores for conflicts

    threshold = 0.6
    y_pred = (y_scores > threshold).astype(int)

    # Compute metrics
    metrics = evaluator.compute_metrics(y_true, y_pred, y_scores)

    # Generate report
    evaluator.generate_report(
        region_name='test_region',
        metrics=metrics,
        dates=dates,
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        threshold=threshold
    )

    print("Evaluation report generated successfully!")
