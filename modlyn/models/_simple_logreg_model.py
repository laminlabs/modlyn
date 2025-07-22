from __future__ import annotations

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torchmetrics import Accuracy, F1Score, MetricCollection


class SimpleLogReg(L.LightningModule):
    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.linear = torch.nn.Linear(n_genes, n_classes)

        metrics = MetricCollection(
            [
                F1Score(num_classes=n_classes, average="macro", task="multiclass"),
                Accuracy(num_classes=n_classes, task="multiclass"),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        # Add batch-level loss tracking
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_steps: list[int] = []  # Track global steps for plotting
        self.val_steps: list[int] = []

    def forward(self, inputs):
        return self.linear(inputs)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, targets)

        # Store batch-level loss
        self.train_losses.append(loss.item())
        self.train_steps.append(self.global_step)

        self.log("train_loss", loss)
        metrics = self.train_metrics(preds, targets)
        self.log_dict(metrics)
        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, targets)

        # Store batch-level validation loss
        self.val_losses.append(loss.item())
        self.val_steps.append(self.global_step)

        self.log("val_loss", loss)
        metrics = self.val_metrics(preds, targets)
        self.log_dict(metrics)

    def on_validation_epoch_end(self) -> None:
        self.val_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def plot_losses(self, figsize=(15, 6)):
        """Plot training and validation losses over training steps."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Training loss per batch
        if self.train_losses and self.train_steps:
            axes[0].plot(
                self.train_steps, self.train_losses, "b-", linewidth=1, alpha=0.7
            )
            axes[0].set_xlabel("Training Steps")
            axes[0].set_ylabel("Training Loss")
            axes[0].set_title("Training Loss Over Steps (Batch Level)")
            axes[0].grid(True, alpha=0.3)

        # Validation loss per batch
        if self.val_losses and self.val_steps:
            axes[1].plot(self.val_steps, self.val_losses, "r-", linewidth=1, alpha=0.7)
            axes[1].set_xlabel("Validation Steps")
            axes[1].set_ylabel("Validation Loss")
            axes[1].set_title("Validation Loss Over Steps (Batch Level)")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        if self.train_losses:
            print(f"Final training loss: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"Final validation loss: {self.val_losses[-1]:.4f}")

    def plot_classification_report(self, adata):
        # Get predictions on training data
        self.eval()
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            logits = self(X_tensor)
            y_pred = torch.argmax(logits, dim=1).numpy()

        # Prepare true labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(adata.obs["cell_line"])

        # Overall F1
        f1 = f1_score(y_encoded, y_pred, average="weighted")

        print(f"Weighted F1: {f1:.3f}")

        # Get per-class metrics
        report = classification_report(
            y_encoded, y_pred, target_names=le.classes_, output_dict=True
        )
        class_recalls = [report[class_name]["recall"] for class_name in le.classes_]
        class_precisions = [
            report[class_name]["precision"] for class_name in le.classes_
        ]
        class_f1s = [report[class_name]["f1-score"] for class_name in le.classes_]

        # Random baseline
        n_classes = len(le.classes_)
        random_baseline = [1 / n_classes] * n_classes

        # Performance metrics plot
        x = np.arange(len(le.classes_))
        width = 0.2

        plt.figure(figsize=(12, 6))
        plt.bar(x - 1.5 * width, class_recalls, width, label="Recall", alpha=0.8)
        plt.bar(x - 0.5 * width, class_precisions, width, label="Precision", alpha=0.8)
        plt.bar(x + 0.5 * width, class_f1s, width, label="F1 Score", alpha=0.8)
        plt.bar(
            x + 1.5 * width, random_baseline, width, label="Random Baseline", alpha=0.8
        )

        plt.xlabel("Cell Line")
        plt.ylabel("Score")
        plt.title("Performance by Cell Line")
        plt.xticks(x, le.classes_, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
