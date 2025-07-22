from __future__ import annotations

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
        else:
            axes[0].text(
                0.5,
                0.5,
                "No training loss data available",
                transform=axes[0].transAxes,
                ha="center",
                va="center",
            )
            axes[0].set_title("Training Loss - No Data")

        # Validation loss per batch
        if self.val_losses and self.val_steps:
            axes[1].plot(self.val_steps, self.val_losses, "r-", linewidth=1, alpha=0.7)
            axes[1].set_xlabel("Validation Steps")
            axes[1].set_ylabel("Validation Loss")
            axes[1].set_title("Validation Loss Over Steps (Batch Level)")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No validation loss data available",
                transform=axes[1].transAxes,
                ha="center",
                va="center",
            )
            axes[1].set_title("Validation Loss - No Data")

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        if self.train_losses:
            print(f"Final training loss: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"Final validation loss: {self.val_losses[-1]:.4f}")
