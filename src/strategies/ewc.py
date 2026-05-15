from .base import ContinualLearningStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F

# the implementation follows:
#   - Overcoming catastrophic forgetting in neural networks [Kirkpatrick et al. (2017)] (https://www.pnas.org/doi/full/10.1073/pnas.1611835114)
#   - Continual Learning – A Deep Dive Into Elastic Weight Consolidation Loss [Kravets (2024)] (https://towardsdatascience.com/continual-learning-a-deep-dive-into-elastic-weight-consolidation-loss-7cda4a2d058c/)


class EWCStrategy(ContinualLearningStrategy):
    """Regularization based strategy using elastic weight consolidation (EWC)."""

    def __init__(self, reg_param: float = 10000):
        self.reg_param = reg_param  # control how much we want to protect old parameters
        self.tasks = []
        self.prev_params = {}
        self.fisher = {}

    def _ewc_loss(self, model: nn.Module, fisher, prev_params, reg_param: float):
        """Compute the EWC regularization term"""
        losses = []
        for name, param in model.named_parameters():
            if name in prev_params:
                p_i = prev_params[name]
                F_i = fisher[name]
                losses.append((F_i * (param - p_i) ** 2).sum())
        return (reg_param / 2) * sum(losses)

    def _estimate_fisher(self, model: nn.Module, train_loader):
        """Compute the fisher information matrix required for the EWC loss."""
        device = next(model.parameters()).device
        fisher = {}

        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param)

        model.eval()
        for i, (input, target) in enumerate(train_loader):
            model.zero_grad()
            output = model(input.to(device))
            target = target.to(device)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()

            # this is an estimation of the fisher information matrix, calculating
            # via the outer product uses too much memory (around 40 GB for MNIST MLP)
            for n, p in model.named_parameters():
                fisher[n].data += p.grad.data**2 / len(train_loader)

        fisher = {name: param for name, param in fisher.items()}
        return fisher

    def before_task(self, model: nn.Module, task_id: str, num_tasks: int) -> None:
        """No preparation needed for EWC."""
        pass

    def compute_loss(self, model: nn.Module, images, outputs, targets, task_id: str):
        """Cross-entropy loss with an additional regularization term for EWC."""
        if len(self.tasks) == 0:
            # first task without regularization
            return F.cross_entropy(outputs, targets)
        else:
            return F.cross_entropy(outputs, targets) + self._ewc_loss(
                model, self.fisher, self.prev_params, self.reg_param
            )

    def after_backward(self, model: nn.Module, task_id: str) -> None:
        """No gradient post-processing needed for EWC strategy."""
        pass

    def after_task(self, model: nn.Module, task_id: str, train_loader) -> None:
        """We need to store fisher information and current weights EWC regularization."""
        self.tasks.append(task_id)

        # we only need to store the current parameters, because they contain
        # information on all previous tasks (same for fisher information matrix)
        current_params = {}
        for name, param in model.named_parameters():
            current_params[name] = param.data.clone()
        self.prev_params = current_params

        fisher = self._estimate_fisher(model, train_loader)
        self.fisher = fisher
