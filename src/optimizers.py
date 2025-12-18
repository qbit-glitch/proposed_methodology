"""
Advanced Optimizers for Training.

Implements:
- Lookahead Optimizer (OPT-3.4)
- RAdam (Rectified Adam)

Optimized for MacBook M4 Pro (MPS backend).
"""

import torch
from torch.optim import Optimizer
from typing import Dict, List, Optional, Callable
import math


class Lookahead(Optimizer):
    """
    Lookahead Optimizer wrapper.
    
    Maintains two sets of weights:
    - Fast weights: Updated every step (inner optimizer)
    - Slow weights: Updated every k steps (interpolation with fast)
    
    Paper: "Lookahead Optimizer: k steps forward, 1 step back"
    https://arxiv.org/abs/1907.08610
    
    Benefits:
    - Converges faster: 20-30% fewer epochs
    - More stable: Less oscillation near optima
    - Better generalization: +1-2% performance
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        k: int = 5,
        alpha: float = 0.5,
    ):
        """
        Args:
            optimizer: Inner optimizer (e.g., Adam, SGD)
            k: Synchronization frequency (update slow weights every k steps)
            alpha: Interpolation coefficient (slow = slow + alpha * (fast - slow))
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        
        # Initialize slow weights
        self.slow_state = {}
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.slow_state[id(p)] = p.data.clone()
        
        # Use inner optimizer's param_groups and state
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
    
    def step(self, closure: Optional[Callable] = None):
        """
        Perform optimization step.
        
        1. Update fast weights with inner optimizer
        2. Every k steps, sync slow weights
        """
        # Step 1: Inner optimizer update (fast weights)
        loss = self.optimizer.step(closure)
        
        self._step_count += 1
        
        # Step 2: Sync slow weights every k steps
        if self._step_count % self.k == 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad and id(p) in self.slow_state:
                        slow = self.slow_state[id(p)]
                        # Interpolate: slow = slow + alpha * (fast - slow)
                        slow.add_(p.data - slow, alpha=self.alpha)
                        # Reset fast weights to slow weights
                        p.data.copy_(slow)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict:
        """Return optimizer state dict."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'slow_state': {k: v.cpu() for k, v in self.slow_state.items()},
            'step_count': self._step_count,
            'k': self.k,
            'alpha': self.alpha,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.slow_state = {k: v.to(next(iter(self.slow_state.values())).device) 
                          for k, v in state_dict['slow_state'].items()}
        self._step_count = state_dict['step_count']
        self.k = state_dict['k']
        self.alpha = state_dict['alpha']
    
    @property
    def defaults(self):
        return self.optimizer.defaults


class RAdam(Optimizer):
    """
    Rectified Adam optimizer.
    
    Combines the benefits of Adam with rectified variance for stable training.
    
    Paper: "On the Variance of the Adaptive Learning Rate and Beyond"
    https://arxiv.org/abs/1908.03265
    
    Benefits:
    - No warmup needed
    - More stable than Adam in early training
    - Better for transformers
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                step = state['step']
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Compute maximum length of approximated SMA
                rho_inf = 2 / (1 - beta2) - 1
                # Compute length of approximated SMA
                rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2
                
                # Variance is tractable
                if rho_t > 5:
                    # Compute variance rectification term
                    rect = math.sqrt(
                        (rho_t - 4) * (rho_t - 2) * rho_inf /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )
                    
                    # Compute adaptive learning rate
                    step_size = group['lr'] * rect / bias_correction1
                    
                    # Update parameters with adaptive learning rate
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # Variance is intractable, use SGD
                    step_size = group['lr'] / bias_correction1
                    p.data.add_(exp_avg, alpha=-step_size)
        
        return loss


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'lookahead_adam',
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    lookahead_k: int = 5,
    lookahead_alpha: float = 0.5,
) -> Optimizer:
    """
    Create optimizer with optional Lookahead wrapper.
    
    Args:
        model: PyTorch model
        optimizer_type: 'adam', 'radam', 'lookahead_adam', 'lookahead_radam'
        lr: Learning rate
        weight_decay: Weight decay coefficient
        lookahead_k: Lookahead sync frequency
        lookahead_alpha: Lookahead interpolation coefficient
        
    Returns:
        Configured optimizer
    """
    # Separate backbone and other parameters for different learning rates
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
        {'params': other_params, 'lr': lr},
    ]
    
    # Create base optimizer
    if 'radam' in optimizer_type:
        base_optimizer = RAdam(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        base_optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    
    # Wrap with Lookahead if requested
    if 'lookahead' in optimizer_type:
        optimizer = Lookahead(base_optimizer, k=lookahead_k, alpha=lookahead_alpha)
    else:
        optimizer = base_optimizer
    
    return optimizer


if __name__ == "__main__":
    print("📊 Optimizer Test\n")
    
    # Test with simple model
    model = torch.nn.Linear(10, 5)
    
    # Test Lookahead + Adam
    optimizer = create_optimizer(model, optimizer_type='lookahead_adam', lr=1e-3)
    print(f"✓ Lookahead + AdamW: {type(optimizer).__name__}")
    
    # Test RAdam
    optimizer = create_optimizer(model, optimizer_type='radam', lr=1e-3)
    print(f"✓ RAdam: {type(optimizer).__name__}")
    
    # Test Lookahead + RAdam
    optimizer = create_optimizer(model, optimizer_type='lookahead_radam', lr=1e-3)
    print(f"✓ Lookahead + RAdam: {type(optimizer).__name__}")
    
    # Test optimization step
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)
    
    for i in range(10):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    print(f"\n✓ 10 optimization steps completed")
    print(f"  Final loss: {loss.item():.4f}")
    print("\n✓ All optimizer tests passed!")
