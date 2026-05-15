from typing import Dict, List
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CLMetrics:
    """Tracks metrics from the GEM paper (Lopez-Paz et al., 2017).
    
    Collects Matrix R: accuracy on task j after training on task i.
    Computes Backward Transfer (BWT) and Average Accuracy (ACC) metrics.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        # Matrix R with entry R[i,j] = accuracy on task j after training on task i
        self.matrix_R: Dict[str, Dict[str, float]] = {}
        
        # Task order for indexing
        self.task_order: List[str] = []
        
        # Current task being trained
        self.current_task: str = None
        
        # Backward Transfer (BWT) - computed after last task
        self.bwt: float = 0.0

        # Average Accuracy (ACC) - computed after last task
        self.acc: float = 0.0

    def reset(self, task_id: str) -> None:
        """Initialize metrics for a new task.
        
        Args:
            task_id: Task identifier (e.g., 'A', 'B', 'C')
        """
        self.current_task = task_id
        if task_id not in self.matrix_R:
            self.matrix_R[task_id] = {}
            self.task_order.append(task_id)
    
    def record_accuracy_value(self, test_task_id: str, accuracy: float) -> None:
        """Record accuracy value for a test task after training on current task.
        
        Args:
            test_task_id: Task that was evaluated on
            accuracy: The computed accuracy value
        """
        self.matrix_R[self.current_task][test_task_id] = accuracy

    def compute_bwt(self) -> None:
        """Compute Backward Transfer (BWT) metric.
        
        BWT measures average accuracy loss on old tasks after learning new tasks.
        """
        T = len(self.task_order)
        if T < 2:
            self.bwt = 0.0
            return
        
        last_task = self.task_order[-1]
        sum_bwt = 0.0
        for i in range(T - 1):
            task_i = self.task_order[i]
            sum_bwt += self.matrix_R[last_task][task_i] - self.matrix_R[task_i][task_i]
        
        self.bwt = sum_bwt / (T - 1)
    
    def compute_acc(self) -> None:
        """Compute Average Accuracy (ACC) metric.
        
        ACC measures final accuracy across all tasks after learning all tasks.
        """
        T = len(self.task_order)
        if T == 0:
            self.acc = 0.0
            return
        
        last_task = self.task_order[-1]
        sum_acc = 0.0
        for task_id in self.task_order:
            sum_acc += self.matrix_R[last_task][task_id]
        
        self.acc = sum_acc / T

    def print_summary(self) -> None:
        """Print metrics summary to configured logger."""
        logger.info("\n" + "=" * 60)
        logger.info("CONTINUAL LEARNING METRICS")
        logger.info("=" * 60)
        
        # Print Matrix R
        logger.info("\nMatrix R (accuracy on task j after training on task i):")
        logger.info("-" * 60)
        header = "Trained→Test\t" + "\t".join(self.task_order)
        logger.info(header)
        for trained_task in self.task_order:
            row = f"{trained_task}\t\t"
            for test_task in self.task_order:
                if test_task in self.matrix_R[trained_task]:
                    acc = self.matrix_R[trained_task][test_task]
                    row += f"{acc:.4f}\t"
                else:
                    row += f"  -  \t"
            logger.info(row)
        
        # Compute and print metrics
        self.compute_bwt()
        self.compute_acc()
        logger.info(f"\nBackward Transfer (BWT): {self.bwt:+.4f}")
        logger.info(f"Average Accuracy (ACC):  {self.acc:.4f}")
        logger.info("=" * 60)
    
    def save(self, filepath: str) -> None:
        """Save metrics to JSON file.
        
        Args:
            filepath: Path to save metrics to
        """
        self.compute_bwt()
        self.compute_acc()
        
        metrics_data = {
            'matrix_R': self.matrix_R,
            'task_order': self.task_order,
            'bwt': self.bwt,
            'acc': self.acc
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load metrics from JSON file.
        
        Args:
            filepath: Path to load metrics from
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {filepath}")
        
        with open(path, 'r') as f:
            metrics_data = json.load(f)
        
        self.matrix_R = metrics_data.get('matrix_R', {})
        self.task_order = metrics_data.get('task_order', [])
        self.bwt = metrics_data.get('bwt', 0.0)
        self.acc = metrics_data.get('acc', 0.0)
    



    