from .base import ContinualLearningStrategy
from typing import Dict, List
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import quadprog

# the implementation follows:
#   - Gradient Episodic Memory for Continual Learning [Lopez-Paz et al. (2017)] (http://arxiv.org/abs/1706.08840)

class GEMStrategy(ContinualLearningStrategy):
    """Replay based strategy using Gradient Episodic Memory (GEM)."""

    def __init__(self, mem_size: int = 5120, margin: float = 0.5):
        self.mem_size: int = mem_size
        self.margin: float = margin
        self.n_memories: int = None
        # using a deque as an alternative to a ringbuffer
        self.mem: Dict[str, deque] = {}
        self.task_order: List[str] = []

    def _project2cone2(self, gradient, memories, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        # This method is adapted from facebookresearch/GradientEpisodicMemory (GEM) 
        # and slightly changed by GenAI (GitHub Copilot) to work in this project,
        # CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
        # Source: https://github.com/facebookresearch/GradientEpisodicMemory (file: model/gem, commit: 34c6b8e)

        if memories.dim() != 2:
            raise ValueError(f"Expected memories to be 2D (t, p), got shape {tuple(memories.shape)}")

        memories_np = memories.detach().cpu().double().numpy()
        gradient_np = gradient.detach().cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.from_numpy(x).to(gradient.device, dtype=gradient.dtype).view(-1, 1))

    def before_task(self, model: nn.Module, task_id: str, num_tasks: int) -> None:
        """Compute the size of memory per task with respect to whole memory size and number of tasks an."""
        # track task order for loss calculation on memories
        if task_id not in self.task_order:
            self.task_order.append(task_id)

        self.n_memories = int(self.mem_size / num_tasks)
        self.mem[task_id] = deque(maxlen=self.n_memories)

    
    def _trainable_params(self, model: nn.Module):
        """Return model parameters that require gradients."""
        return [p for p in model.parameters() if p.requires_grad]

    def _flatten_grads(self, grads, params):
        """Flatten a gradient list into one vector, replacing missing grads with zeros."""
        flat = []
        for p, g in zip(params, grads):
            if g is None:
                flat.append(torch.zeros_like(p).view(-1))
            else:
                flat.append(g.detach().view(-1))
        return torch.cat(flat)

    def _flatten_current_grads(self, params):
        """Flatten current ``.grad`` tensors from parameters into one vector."""
        flat = []
        for p in params:
            if p.grad is None:
                flat.append(torch.zeros_like(p).view(-1))
            else:
                flat.append(p.grad.detach().view(-1))
        return torch.cat(flat)

    def _assign_flat_grads(self, params, flat_grad):
        """Write a flat gradient vector back into parameter ``.grad`` buffers."""
        # This method was written with strong support from GenAI (GitHub Copilot) 
        pointer = 0
        for p in params:
            numel = p.numel()
            grad_slice = flat_grad[pointer:pointer + numel].view_as(p)
            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)
            pointer += numel

    def _grad_on_memory(self, model, task_id):
        """Compute flattened gradient on episodic memory of one previous task."""
        if len(self.mem[task_id]) == 0:
            return None

        device = next(model.parameters()).device
        xs, ys = zip(*self.mem[task_id])
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)

        # Use eval mode for stable GEM constraints (no dropout / batchnorm as in train mode),
        # then restore the previous mode for normal training.
        was_training = model.training
        model.eval()
        try:
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)

            params = self._trainable_params(model)
            grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
            return self._flatten_grads(grads, params)
        finally:
            if was_training:
                model.train()

    def _previous_task_ids(self, task_id):
        """Return all task ids that were seen before the current task."""
        t = self.task_order.index(task_id)
        return self.task_order[:t]

    def compute_loss(self, model: nn.Module, images, outputs, targets, task_id: str):
        """Standard task loss while filling episodic memory for GEM."""
        with torch.no_grad():
            for x_i, y_i in zip(images, targets):
                self.mem[task_id].append((
                    x_i.detach().cpu().clone(),
                    y_i.detach().cpu().clone(),
                ))

        return F.cross_entropy(outputs, targets)

    def after_backward(self, model: nn.Module, task_id: str) -> None:
        """Project current gradient in such a way that GEM-constraint is fulfilled for every past task."""
        prev_ids = self._previous_task_ids(task_id)
        if not prev_ids:
            return

        params = self._trainable_params(model)
        # retrieve current gradient
        g = self._flatten_current_grads(params)

        g_list = []
        # compute loss-gradient for every past task with corresponding memories
        for old_id in prev_ids:
            gk = self._grad_on_memory(model, old_id)
            if gk is not None:
                g_list.append(gk.to(g.device))

        if not g_list:
            return

        # vectorized computation of scalar products <g,g_k>
        memories = torch.stack(g_list)
        dot_products = torch.mv(memories, g)
        if torch.all(dot_products >= 0):
            return

        # project the proposed gradient
        projected = g.clone().view(-1, 1)
        self._project2cone2(projected, memories, margin=self.margin)
        self._assign_flat_grads(params, projected.view(-1))

    def after_task(self, model: nn.Module, task_id: str, train_loader) -> None:
        """No post-task bookkeeping required for this GEM variant."""
        pass
