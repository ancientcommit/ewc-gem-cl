"""Custom metrics collection for continual learning.

Tracks Matrix R (accuracy on all tasks), Backward Transfer (BWT), and per-task accuracy.
Insprired by / based on the GEM paper by Lopez-Paz et al. (2017)
"""

from .clmetrics import CLMetrics

__all__ = ['CLMetrics']
