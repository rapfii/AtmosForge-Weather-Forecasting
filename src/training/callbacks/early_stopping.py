"""Early stopping callback for training.

Monitors validation loss and stops training when no improvement
is seen for `patience` consecutive epochs.
"""

from __future__ import annotations

from src.utils.logger import setup_logger

logger = setup_logger("atmosforge.training.early_stopping")


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Monitors a metric (typically val_loss) and triggers a stop when
    no improvement is seen for `patience` consecutive epochs.

    Args:
        patience: Number of epochs to wait for improvement (default: 10).
        min_delta: Minimum change to qualify as improvement (default: 1e-6).
        mode: 'min' for metrics to minimize, 'max' to maximize.

    Example:
        >>> es = EarlyStopping(patience=10)
        >>> for epoch in range(max_epochs):
        ...     val_loss = validate()
        ...     if es.step(val_loss):
        ...         print("Early stopping triggered!")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter: int = 0
        self.best_score: float | None = None
        self.should_stop: bool = False

        if mode == "min":
            self._is_improvement = lambda current, best: current < best - min_delta
        elif mode == "max":
            self._is_improvement = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def step(self, metric_value: float) -> bool:
        """Check if training should stop.

        Args:
            metric_value: Current metric value to check.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = metric_value
            return False

        if self._is_improvement(metric_value, self.best_score):
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"EarlyStopping: no improvement for {self.patience} epochs "
                    f"(best={self.best_score:.6f})"
                )
                return True

        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
