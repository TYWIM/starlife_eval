"""v2 task-matrix benchmark: model-agnostic tasks, deterministic scoring, statistics."""

from .schema import CAPABILITIES, CAPABILITY_NAMES, Prediction, Task, load_predictions, load_tasks, write_predictions, write_tasks
from .scoring import bootstrap_ci, is_correct, mcnemar, score_run

__all__ = [
    "CAPABILITIES", "CAPABILITY_NAMES", "Prediction", "Task", "load_predictions",
    "load_tasks", "write_predictions", "write_tasks", "bootstrap_ci", "is_correct",
    "mcnemar", "score_run",
]
