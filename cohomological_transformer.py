#!/usr/bin/env python3
#
# A single, self-contained library that merges
# nearly all code from the discussion about "Cohomological Transformers."
# 
# WARNING: This code is for demonstration only. Many classes reference
# advanced math structures in extremely speculative ways. Actual correctness
# or training stability is not guaranteed.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    Any, Dict, List, Tuple, Optional, Set
)
from collections import defaultdict
import numpy as np

################################################################################
# SVD-BASED UTILS (from conversation about dimension adaptation, SVD, etc.)
################################################################################

class SVDTracker:
    """
    A small helper used in the dimension-analyzing code. For large real models,
    this would be unfeasible, but in toy code we can do it.
    """
    def __init__(self, dim: int):
        self.dim = dim

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """
        For demonstration: flatten x, do SVD, return singular values.
        """
        # flatten everything except last dimension
        B, N, D = x.shape
        mat = x.reshape(B*N, D)
        # we do a small random subsampling if it's too big
        # in a real scenario, we might do random projection or partial SVD
        if mat.shape[0] > 2000:
            mat = mat[torch.randperm(mat.shape[0])[:2000]]
        # compute SVD
        U, s, V = torch.svd(mat)
        return s


################################################################################
# cohomology_core/ExactSequenceModules.py (various expansions)
################################################################################

class ExactSequencePreservingTransition(nn.Module):
    """
    Handles transitions between cohomology levels while preserving exact sequences.
    (From prior code expansions.)
    """
    def __init__(self, in_dim: int, out_dim: int, sequence_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sequence_dim = sequence_dim
        
        self.sequence_projector = nn.Linear(in_dim, sequence_dim)
        self.main_projector = nn.Linear(in_dim, out_dim - sequence_dim)
        self.exactness_mask = nn.Parameter(torch.ones(sequence_dim))
        
        self.exactness_gate = nn.Sequential(
            nn.Linear(in_dim, sequence_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_component = self.sequence_projector(x)
        main_component = self.main_projector(x)
        
        gate = self.exactness_gate(x)
        seq_component = seq_component * gate * self.exactness_mask
        
        return torch.cat([seq_component, main_component], dim=-1)


class SequenceTracker(nn.Module):
    """
    Tracks and manages sequence-preserving components of the representation.
    (From prior code expansions.)
    """
    def __init__(self, dim: int, sequence_dim: int):
        super().__init__()
        self.dim = dim
        self.sequence_dim = sequence_dim
        self.seq_projector = nn.Linear(dim, sequence_dim)
        self.other_projector = nn.Linear(dim, dim - sequence_dim)
        self.sequence_mask = nn.Parameter(torch.ones(sequence_dim))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_component = self.seq_projector(x) * self.sequence_mask
        other_component = self.other_projector(x)
        return seq_component, other_component
    
    def combine(self, seq: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            seq * self.sequence_mask,
            other
        ], dim=-1)


class ExactSequenceAttention(nn.Module):
    """
    The 'final' version of the exact-sequence attention from the "greatest hits."
    Mixed ideas from multiple expansions.
    """
    def __init__(self, dim: int, num_heads: int = 8, sequence_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sequence_dim = sequence_dim
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)

        self.sequence_projector = nn.Linear(dim, sequence_dim)
        self.kernel_checker = nn.Linear(sequence_dim, sequence_dim)
        self.image_checker = nn.Linear(sequence_dim, sequence_dim)

        # For "exactness" gating
        self.sequence_mask = nn.Parameter(torch.ones(num_heads, sequence_dim))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Normal attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)

        # "Exact-sequence" augmentation
        seq_q = self.sequence_projector(q)
        seq_k = self.sequence_projector(k)
        ker = self.kernel_checker(seq_q)
        img = self.image_checker(seq_k)
        seq_scores = torch.matmul(ker, img.transpose(-2, -1)) / math.sqrt(self.sequence_dim)
        seq_scores = seq_scores * self.sequence_mask[None, :, None]
        seq_weights = F.softmax(seq_scores, dim=-1)

        final_weights = 0.5 * (attn + seq_weights)
        y = (final_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(y)

        metrics = {
            'attn_norm': attn.norm().item(),
            'seq_attn_norm': seq_weights.norm().item()
        }
        return out, metrics


class ExactSequenceLoss(nn.Module):
    """
    A toy loss that penalizes 'violations' of an abstract exactness
    condition. (Final 'consolidated' version.)
    """
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self,
                representation_list: List[torch.Tensor],
                base_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if len(representation_list) < 2:
            return base_loss, {}

        total_violation = 0.0
        violations = []
        for i in range(len(representation_list) - 1):
            hi = representation_list[i]
            hi1 = representation_list[i+1]
            violation = torch.norm(torch.matmul(hi1, hi.transpose(-2, -1)))
            total_violation += violation
            violations.append(violation.item())

        total_loss = base_loss + self.weight * total_violation
        return total_loss, {
            'exact_violations': violations,
            'exact_loss': total_violation.item(),
            'base_loss': base_loss.item(),
            'total_loss': total_loss.item()
        }


################################################################################
# cohomology_core/SpectralSequenceModules.py
################################################################################

class LocalAttention(nn.Module):
    """
    A local attention mechanism. (From the expansions.)
    """
    def __init__(self, dim: int, heads: int, window_size: int = 16):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # We'll just do a very naive local mask: if |i-j| > window_size => masked.
        idxs = torch.arange(N, device=x.device).unsqueeze(0).unsqueeze(0)
        dist = (idxs - idxs.transpose(-1,-2)).abs()
        local_mask = (dist <= self.window_size).float()  # 1 if within window

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(local_mask==0, float('-inf'))
        if mask is not None:
            attn = attn + (mask - 1)*1e9  # or something

        attn = F.softmax(attn, dim=-1)
        x_out = (attn @ v).transpose(1,2).reshape(B, N, C)
        x_out = self.proj(x_out)
        return x_out


class SpectralSequenceLayer(nn.Module):
    """
    Maintains a spectral-sequence structure across multiple 'levels'.
    (A final consolidated version from expansions.)
    """
    def __init__(self, dim: int, num_levels: int = 3, filtration_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.filtration_dim = filtration_dim

        self.filtrations = nn.ModuleList([
            nn.Linear(dim, filtration_dim) for _ in range(num_levels)
        ])
        self.differentials = nn.ModuleList([
            nn.Linear(filtration_dim, filtration_dim) for _ in range(num_levels-1)
        ])
        self.exactness_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(2*filtration_dim, filtration_dim), nn.Sigmoid())
            for _ in range(num_levels-1)
        ])

    def compute_differential(self, level: int, x: torch.Tensor) -> torch.Tensor:
        return self.differentials[level](x)
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        device = x.device
        levels = []
        exactness_violations = torch.tensor(0.0, device=device)

        # filter input
        for i in range(self.num_levels):
            Li = self.filtrations[i](x)
            levels.append(Li)

            if 0 < i < (self.num_levels-1):
                # measure exactness
                prev_img = self.compute_differential(i-1, levels[i-1])
                next_ker = self.compute_differential(i, Li)
                violation = torch.norm(torch.matmul(next_ker, prev_img.transpose(-2,-1)))
                exactness_violations += violation

                gate = self.exactness_gates[i-1](torch.cat([levels[i-1], Li], dim=-1))
                levels[i] = Li * gate

        return levels, exactness_violations


class LocalGlobalBlock(nn.Module):
    """
    A small “local-to-global” block that merges local conv + global attn.
    (Final consolidated version.)
    """
    def __init__(self, dim: int, kernel_size: int = 3, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.local_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=1)
        self.global_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim*2, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        B, N, C = x.shape
        x_local = x.transpose(1,2)
        x_local = self.local_conv(x_local)
        x_local = x_local.transpose(1,2)

        x_global, _ = self.global_attn(x, x, x, need_weights=False, attn_mask=mask)
        merged = torch.cat([x_local, x_global], dim=-1)
        out = self.proj(merged)
        return out, {}


class SpectralSequenceLoss(nn.Module):
    """
    A toy loss that tries to ensure local/global are consistent.
    """
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, local_repr: torch.Tensor, global_repr: torch.Tensor, base_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        mismatch = torch.norm(local_repr - global_repr)
        total_loss = base_loss + self.weight * mismatch
        return total_loss, {
            'spectral_loss': mismatch.item(),
            'base_loss': base_loss.item(),
            'total_loss': total_loss.item()
        }


################################################################################
# Additional EXACT SEQUENCE / SPECTRAL CROSS modules (some from expansions)
################################################################################

class ExactSequenceCell(nn.Module):
    """
    A 'cell' that tries to maintain exact sequences in an LSTM-like manner.
    (From the expansions.)
    """
    def __init__(self, dim: int, sequence_dim: int):
        super().__init__()
        self.dim = dim
        self.sequence_dim = sequence_dim

        self.sequence_gate = nn.Linear(dim + sequence_dim, sequence_dim)
        self.update_gate = nn.Linear(dim + sequence_dim, sequence_dim)
        self.output_gate = nn.Linear(dim + sequence_dim, sequence_dim)

        self.sequence_transform = nn.Linear(sequence_dim, sequence_dim)
        self.exactness_check = nn.Linear(2*sequence_dim, 1)

    def forward(self, x: torch.Tensor, prev_sequence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_sequence is None:
            prev_sequence = torch.zeros(x.shape[0], self.sequence_dim, device=x.device)

        combined = torch.cat([x, prev_sequence], dim=-1)
        seq_gate = torch.sigmoid(self.sequence_gate(combined))
        upd_gate = torch.sigmoid(self.update_gate(combined))
        out_gate = torch.sigmoid(self.output_gate(combined))

        seq_candidate = self.sequence_transform(prev_sequence)
        new_seq = (upd_gate * seq_candidate + (1 - upd_gate)*prev_sequence)

        exact_val = torch.sigmoid(self.exactness_check(torch.cat([new_seq, prev_sequence], dim=-1)))
        final_seq = seq_gate*new_seq*exact_val + (1 - seq_gate*exact_val)*prev_sequence
        output = out_gate * final_seq
        return output, final_seq


################################################################################
# pruning/CriticalCircuitDetection.py
################################################################################

class CriticalSequenceDetector(nn.Module):
    """
    Minimal "circuit" detection: measure norm, if above threshold => critical.
    (Final consolidated version.)
    """
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.detected_count = 0

    def forward(self, x: torch.Tensor) -> Tuple[bool, Dict]:
        norm_val = x.norm().item()
        is_critical = (norm_val > self.threshold)
        if is_critical:
            self.detected_count += 1
        return is_critical, {'act_norm': norm_val, 'critical_count': self.detected_count}


class CircuitPreservationPruner:
    """
    A trivial pruning approach, using param norm as a proxy for importance.
    (Final consolidated version.)
    """
    def __init__(self, model: nn.Module, prune_ratio: float = 0.5):
        self.model = model
        self.prune_ratio = prune_ratio
        self.param_scores = {}

    def compute_param_scores(self):
        for name, p in self.model.named_parameters():
            self.param_scores[name] = p.data.norm().item()

    def prune(self):
        self.compute_param_scores()
        all_scores = sorted(self.param_scores.values())
        cutoff_idx = int(len(all_scores)*self.prune_ratio)
        cutoff = all_scores[cutoff_idx] if cutoff_idx < len(all_scores) else 0.0

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if self.param_scores[name] < cutoff:
                    p.data.mul_(0.0)


################################################################################
# losses/CohomologicalObjective.py
################################################################################

class CohomologicalTrainingObjective:
    """
    A final "meta-loss" that combines a base task loss with
    exact-sequence and spectral-sequence losses.
    """
    def __init__(self, exact_loss_weight: float = 0.1, spectral_loss_weight: float = 0.1):
        self.exact_loss_fn = ExactSequenceLoss(exact_loss_weight)
        self.spectral_loss_fn = SpectralSequenceLoss(spectral_loss_weight)

    def __call__(self, outputs: Dict[str, torch.Tensor], base_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:

        total_loss = base_loss
        logs: Dict[str, Any] = {}

        if 'repr_list' in outputs:
            exact_combined, exact_logs = self.exact_loss_fn(outputs['repr_list'], base_loss)
            total_loss = exact_combined
            logs['exact_loss'] = exact_logs

        if 'local_repr' in outputs and 'global_repr' in outputs:
            # re-add base_loss
            spectral_combined, spectral_logs = self.spectral_loss_fn(
                outputs['local_repr'], outputs['global_repr'], base_loss
            )
            total_loss = spectral_combined
            logs['spectral_loss'] = spectral_logs

        logs['final_total_loss'] = total_loss.item()
        return total_loss, logs


################################################################################
# optim/ExactSequenceOptimizer.py
################################################################################

class ExactSequenceOptimizer(torch.optim.Adam):
    """
    A demonstration of an optimizer that presumably tries
    to preserve "exact sequences." It's basically Adam.
    """
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr=lr)
    # If we wanted, we could override step().


################################################################################
# Additional code from expansions (like spectral LR schedulers, etc.)
# We'll include them for completeness, though we won't use them in the final demo
################################################################################

class SpectralLearningRateScheduler:
    """
    Adjusts learning rates based on spectral sequence convergence.
    (Large from expansions, minimal usage in final script.)
    """
    def __init__(self, base_lr: float = 1e-3, num_levels: int = 3, warmup_steps: int = 1000):
        self.base_lr = base_lr
        self.num_levels = num_levels
        self.warmup_steps = warmup_steps
        self.level_convergence = [False]*num_levels
        self.convergence_scores = defaultdict(list)
        self.lr_multipliers = torch.ones(num_levels)

    def compute_level_convergence(self, level: int, page_outputs: List[torch.Tensor]) -> float:
        if level >= len(page_outputs)-1:
            return 1.0
        current = page_outputs[level]
        nxt = page_outputs[level+1]
        diff = torch.norm(nxt - current)/torch.norm(current)
        return 1.0/(1.0+diff)

    def update_convergence(self, page_outputs: List[torch.Tensor]):
        for level in range(self.num_levels):
            score = self.compute_level_convergence(level, page_outputs)
            self.convergence_scores[level].append(score)
            if len(self.convergence_scores[level])>100:
                recent = self.convergence_scores[level][-100:]
                avg_score = sum(recent)/100
                self.level_convergence[level] = (avg_score>0.95)

    def get_level_lr(self, level: int, step: int) -> float:
        warmup_factor = min(1.0, step/self.warmup_steps)
        lr = self.base_lr * warmup_factor
        for lower_level in range(level):
            if not self.level_convergence[lower_level]:
                lr*=0.1
        return lr*self.lr_multipliers[level].item()

    def step(self, page_outputs: List[torch.Tensor], step: int) -> Dict[int,float]:
        self.update_convergence(page_outputs)
        learning_rates = {
            lvl: self.get_level_lr(lvl,step) for lvl in range(self.num_levels)
        }
        return learning_rates


class ExactSequenceOptimizerAdvanced(torch.optim.Optimizer):
    """
    Another advanced sample from expansions. We won't use it in the final
    demo, but we keep it to unify the code base.
    """
    def __init__(self,
                 params,
                 lr: float=1e-3,
                 betas: Tuple[float,float]=(0.9,0.999),
                 eps: float=1e-8,
                 weight_decay: float=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.sequence_params = defaultdict(list)

    def register_sequence(self, sequence_id: str, param_names: List[str]):
        for n in param_names:
            self.sequence_params[n].append(sequence_id)

    def compute_sequence_update(self, param_name: str, grad: torch.Tensor)->torch.Tensor:
        sequences = self.sequence_params[param_name]
        if not sequences:
            return grad
        # Suppose we average grads across param sets in each sequence
        # TOTALLY hypothetical
        return grad

    def get_param_grad(self, param_name: str)->Optional[torch.Tensor]:
        for group in self.param_groups:
            for p in group['params']:
                if getattr(p, '_param_name', None)==param_name:
                    return p.grad
        return None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = getattr(p,'_param_name',None)
                if param_name is None:
                    grad = p.grad
                else:
                    grad = self.compute_sequence_update(param_name,p.grad)
                # do standard Adam steps...
                # omitted for brevity
        return loss


################################################################################
# A toy model that references the newly integrated modules
################################################################################

class ToyCohomologicalModel(nn.Module):
    """
    A small demonstration model that:
      - uses the final version of ExactSequenceAttention
      - merges local/global
      - outputs classification
    """
    def __init__(self, dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.dim = dim
        self.attn = ExactSequenceAttention(dim, num_heads=4, sequence_dim=16)
        self.locglob = LocalGlobalBlock(dim, kernel_size=3, num_heads=2)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        a_out, _ = self.attn(x, mask=mask)
        lg_out, _ = self.locglob(a_out, mask=mask)
        logits = self.fc(lg_out.mean(dim=1))

        return {
            'repr_list': [x, a_out, lg_out],
            'local_repr': a_out,
            'global_repr': lg_out,
            'logits': logits
        }


################################################################################
# Another bigger "CohomologicalTransformer" from expansions (some repeated ideas).
################################################################################

class CohomologicalTransformer(nn.Module):
    """
    A bigger architecture that tries to incorporate:
      - H0/H1/H2 levels
      - transitions
      - sequence trackers
    Not guaranteed to run well, but included for completeness.
    """
    def __init__(self,
                 dim: int = 512,
                 sequence_dim: int = 64,
                 num_h0_layers: int = 4,
                 num_h1_layers: int = 2,
                 num_h2_layers: int = 1,
                 h0_heads: int = 8,
                 h1_heads: int = 4,
                 h2_heads: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(30000, dim)
        self.h0_layers = nn.ModuleList([
            LocalAttention(dim, h0_heads, window_size=16)
            for _ in range(num_h0_layers)
        ])
        self.h0_to_h1 = ExactSequencePreservingTransition(dim, dim, sequence_dim)
        self.h1_layers = nn.ModuleList([
            LocalAttention(dim, h1_heads, window_size=8)  # or SparseAttn
            for _ in range(num_h1_layers)
        ])
        self.h1_to_h2 = ExactSequencePreservingTransition(dim, dim, sequence_dim)
        self.h2_layers = nn.ModuleList([
            LocalAttention(dim, h2_heads, window_size=4)
            for _ in range(num_h2_layers)
        ])
        self.h0_tracker = SequenceTracker(dim, sequence_dim)
        self.h1_tracker = SequenceTracker(dim, sequence_dim)
        self.h2_tracker = SequenceTracker(dim, sequence_dim)
        self.output = nn.Linear(dim, 30000)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.embedding(x)
        sequence_components = []

        h0_seq, h0_other = self.h0_tracker(x)
        sequence_components.append(h0_seq)

        for layer in self.h0_layers:
            x = layer(x, mask)

        x = self.h0_to_h1(x)
        h1_seq, h1_other = self.h1_tracker(x)
        sequence_components.append(h1_seq)

        for layer in self.h1_layers:
            x = layer(x, mask)

        x = self.h1_to_h2(x)
        h2_seq, h2_other = self.h2_tracker(x)
        sequence_components.append(h2_seq)

        for layer in self.h2_layers:
            x = layer(x, mask)

        out = self.output(x)
        return out, sequence_components


################################################################################
# A "trainer" from expansions, included for completeness
################################################################################

class CohomologicalTrainer:
    """
    Example trainer that uses the big CohomologicalTransformer,
    does circuit detection, etc.
    """
    def __init__(self, model: nn.Module,
                 lr: float=1e-3,
                 sequence_lr: float=1e-4,
                 alpha: float=0.1,
                 beta: float=0.1,
                 device: str='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = ExactSequenceOptimizer(self.model.parameters(), lr=lr)
        self.criterion = CohomologicalTrainingObjective(alpha, beta)

        self.sequence_registry = defaultdict(set)

        self.circuit_detector = CriticalSequenceDetector(threshold=5.0)
        self.circuit_pruner = CircuitPreservationPruner(model, prune_ratio=0.2)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()
        out, seq_comps = self.model(x.to(self.device))
        # We'll treat out as logits for classification
        # random or dummy
        base_loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y.to(self.device).view(-1))

        # build dummy 'outputs' for the cohomological objective
        outputs = {
            'repr_list': seq_comps,
            'local_repr': seq_comps[0] if len(seq_comps) > 0 else out,
            'global_repr': seq_comps[-1] if len(seq_comps) > 0 else out
        }
        total_loss, logs = self.criterion(outputs, base_loss)
        total_loss.backward()
        self.optimizer.step()

        # circuit detection on final seq
        if len(seq_comps)>0:
            final_seq = seq_comps[-1]
            is_critical, c_logs = self.circuit_detector(final_seq)
            if is_critical:
                self.circuit_pruner.prune()
            logs['critical_count'] = c_logs['critical_count']

        return logs


################################################################################
# A minimal demonstration script
################################################################################

def demo_training_step(model, objective, batch_x, batch_y, optimizer):
    optimizer.zero_grad()
    outputs = model(batch_x)
    base_loss = F.cross_entropy(outputs['logits'], batch_y)
    total_loss, logs = objective(outputs, base_loss)
    total_loss.backward()
    optimizer.step()
    return logs

def main_demo():
    torch.manual_seed(0)
    model = ToyCohomologicalModel(dim=32, num_classes=5)
    objective = CohomologicalTrainingObjective(exact_loss_weight=0.01, spectral_loss_weight=0.01)
    optimizer = ExactSequenceOptimizer(model.parameters(), lr=1e-3)
    detector = CriticalSequenceDetector(threshold=5.0)
    pruner = CircuitPreservationPruner(model, prune_ratio=0.2)

    B, N, D = 8, 10, 32
    x = torch.randn(B, N, D)
    y = torch.randint(0, 5, size=(B,))

    for step in range(5):
        logs = demo_training_step(model, objective, x, y, optimizer)
        final_repr = model(x)['repr_list'][-1]
        is_crit, c_logs = detector(final_repr)
        if is_crit:
            pruner.prune()
        print(f"Step {step}: total_loss={logs['final_total_loss']:.4f}  critical_count={c_logs['critical_count']}")

if __name__ == "__main__":
    main_demo()
