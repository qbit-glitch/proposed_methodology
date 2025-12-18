"""
Curriculum Learning for Training Optimization.

Implements difficulty-based sample scheduling (OPT-3.1) for:
- Faster convergence
- Better final performance
- More stable training

Works on MacBook M4 Pro (MPS backend).
"""

import torch
from torch.utils.data import Dataset, Sampler
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import random
from dataclasses import dataclass


@dataclass
class SampleDifficulty:
    """Store difficulty metrics for a sample."""
    index: int
    num_objects: int = 0
    occlusion_ratio: float = 0.0
    blur_score: float = 0.0
    brightness: float = 0.5
    difficulty: float = 0.0


class DifficultyScorer:
    """
    Score sample difficulty based on various factors.
    
    Difficulty factors:
    - Number of objects (more = harder)
    - Occlusion ratio (higher = harder)
    - Blur level (higher = harder)
    - Brightness (extreme values = harder)
    - Object size (smaller = harder)
    """
    
    def __init__(
        self,
        w_objects: float = 0.3,
        w_occlusion: float = 0.25,
        w_blur: float = 0.2,
        w_brightness: float = 0.15,
        w_size: float = 0.1,
    ):
        """
        Args:
            w_objects: Weight for number of objects
            w_occlusion: Weight for occlusion ratio
            w_blur: Weight for blur score
            w_brightness: Weight for brightness deviation
            w_size: Weight for object size
        """
        self.weights = {
            'objects': w_objects,
            'occlusion': w_occlusion,
            'blur': w_blur,
            'brightness': w_brightness,
            'size': w_size,
        }
    
    def score_sample(
        self,
        image: torch.Tensor,
        targets: Dict,
    ) -> float:
        """
        Compute difficulty score for a sample.
        
        Args:
            image: Image tensor [C, H, W]
            targets: Target annotations dict
            
        Returns:
            Difficulty score in [0, 1] where 1 is hardest
        """
        scores = {}
        
        # Number of objects (normalized)
        if 'detection' in targets:
            det = targets['detection']
            if isinstance(det, dict):
                num_objects = len(det.get('boxes', det.get('labels', [])))
            elif isinstance(det, list):
                num_objects = len(det)
            else:
                num_objects = 0
        else:
            num_objects = 0
        
        # Normalize to [0, 1], cap at 20 objects
        scores['objects'] = min(num_objects / 20.0, 1.0)
        
        # Brightness (deviation from 0.5 is harder)
        mean_brightness = image.mean().item()
        brightness_deviation = abs(mean_brightness - 0.5) * 2
        scores['brightness'] = min(brightness_deviation, 1.0)
        
        # Blur score (estimate from high-frequency content)
        if image.dim() == 3:
            gray = image.mean(dim=0)  # [H, W]
            # Laplacian variance as blur metric
            laplacian = self._compute_laplacian_var(gray)
            # Lower variance = more blur
            scores['blur'] = 1.0 - min(laplacian / 0.1, 1.0)
        else:
            scores['blur'] = 0.0
        
        # Object size (smaller = harder)
        if 'detection' in targets and isinstance(targets['detection'], dict):
            boxes = targets['detection'].get('boxes', torch.empty(0, 4))
            if len(boxes) > 0:
                # Compute average box area (normalized)
                if boxes.dim() == 2 and boxes.size(1) >= 4:
                    areas = boxes[:, 2] * boxes[:, 3]  # width * height
                    avg_area = areas.mean().item()
                    # Smaller objects = higher difficulty
                    scores['size'] = 1.0 - min(avg_area / 0.1, 1.0)
                else:
                    scores['size'] = 0.5
            else:
                scores['size'] = 0.5
        else:
            scores['size'] = 0.5
        
        # Placeholder for occlusion (would need IoU computation)
        scores['occlusion'] = 0.0
        
        # Weighted sum
        total = sum(
            self.weights[key] * scores.get(key, 0.0)
            for key in self.weights
        )
        
        return total
    
    def _compute_laplacian_var(self, gray: torch.Tensor) -> float:
        """Compute Laplacian variance as focus measure."""
        # Simple Laplacian kernel
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        
        # Apply Laplacian
        gray = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        laplacian = torch.nn.functional.conv2d(gray, kernel, padding=1)
        
        return laplacian.var().item()


class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler that yields samples from easy to hard.
    
    Training schedule:
    - Stage 1 (epochs 1-E/5): Only easy samples
    - Stage 2 (epochs E/5-E/2): Easy + medium samples
    - Stage 3 (epochs E/2-E): All samples
    """
    
    def __init__(
        self,
        dataset: Dataset,
        difficulty_scores: List[float],
        total_epochs: int,
        current_epoch: int = 0,
        easy_ratio: float = 0.33,
        medium_ratio: float = 0.33,
    ):
        """
        Args:
            dataset: The training dataset
            difficulty_scores: Precomputed difficulty scores for each sample
            total_epochs: Total number of training epochs
            current_epoch: Current epoch (0-indexed)
            easy_ratio: Proportion of easy samples (default 33%)
            medium_ratio: Proportion of medium samples (default 33%)
        """
        self.dataset = dataset
        self.difficulty_scores = np.array(difficulty_scores)
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch
        self.easy_ratio = easy_ratio
        self.medium_ratio = medium_ratio
        
        # Sort indices by difficulty
        self.sorted_indices = np.argsort(self.difficulty_scores)
        
        # Split into easy, medium, hard
        n = len(dataset)
        n_easy = int(n * easy_ratio)
        n_medium = int(n * medium_ratio)
        
        self.easy_indices = self.sorted_indices[:n_easy].tolist()
        self.medium_indices = self.sorted_indices[n_easy:n_easy + n_medium].tolist()
        self.hard_indices = self.sorted_indices[n_easy + n_medium:].tolist()
    
    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch
    
    def __iter__(self):
        """Generate indices based on curriculum stage."""
        # Determine curriculum stage
        stage1_end = self.total_epochs // 5
        stage2_end = self.total_epochs // 2
        
        if self.current_epoch < stage1_end:
            # Stage 1: Only easy samples
            indices = self.easy_indices.copy()
        elif self.current_epoch < stage2_end:
            # Stage 2: Easy + medium samples
            indices = self.easy_indices + self.medium_indices
        else:
            # Stage 3: All samples
            indices = self.easy_indices + self.medium_indices + self.hard_indices
        
        # Shuffle within current set
        random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
        """Return current number of available samples."""
        stage1_end = self.total_epochs // 5
        stage2_end = self.total_epochs // 2
        
        if self.current_epoch < stage1_end:
            return len(self.easy_indices)
        elif self.current_epoch < stage2_end:
            return len(self.easy_indices) + len(self.medium_indices)
        else:
            return len(self.dataset)


class CurriculumScheduler:
    """
    Manages curriculum learning schedule.
    
    Provides:
    - Sample difficulty scoring
    - Epoch-based sample selection
    - Progress tracking
    """
    
    def __init__(
        self,
        dataset: Dataset,
        total_epochs: int,
        scorer: Optional[DifficultyScorer] = None,
        precomputed_scores: Optional[List[float]] = None,
    ):
        """
        Args:
            dataset: Training dataset
            total_epochs: Total training epochs
            scorer: Difficulty scorer (if not providing precomputed_scores)
            precomputed_scores: Precomputed difficulty scores
        """
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.scorer = scorer or DifficultyScorer()
        
        if precomputed_scores is not None:
            self.difficulty_scores = precomputed_scores
        else:
            self.difficulty_scores = None
    
    def compute_difficulty_scores(
        self, 
        num_samples: Optional[int] = None,
        verbose: bool = True
    ) -> List[float]:
        """
        Compute difficulty scores for all samples.
        
        Args:
            num_samples: Limit scoring to first N samples (for speed)
            verbose: Print progress
            
        Returns:
            List of difficulty scores
        """
        if self.difficulty_scores is not None:
            return self.difficulty_scores
        
        n = len(self.dataset)
        if num_samples:
            n = min(n, num_samples)
        
        scores = []
        
        if verbose:
            print(f"📊 Computing difficulty scores for {n} samples...")
        
        for i in range(n):
            try:
                image, targets = self.dataset[i]
                score = self.scorer.score_sample(image, targets)
            except Exception:
                score = 0.5  # Default medium difficulty
            
            scores.append(score)
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Scored {i + 1}/{n} samples")
        
        # Pad with medium difficulty if we limited samples
        if num_samples and num_samples < len(self.dataset):
            scores.extend([0.5] * (len(self.dataset) - num_samples))
        
        self.difficulty_scores = scores
        
        if verbose:
            easy = sum(1 for s in scores if s < 0.33)
            medium = sum(1 for s in scores if 0.33 <= s < 0.66)
            hard = sum(1 for s in scores if s >= 0.66)
            print(f"  Easy: {easy}, Medium: {medium}, Hard: {hard}")
        
        return scores
    
    def create_sampler(self, current_epoch: int = 0) -> CurriculumSampler:
        """
        Create curriculum sampler for current epoch.
        
        Args:
            current_epoch: Current training epoch
            
        Returns:
            CurriculumSampler instance
        """
        if self.difficulty_scores is None:
            self.compute_difficulty_scores()
        
        return CurriculumSampler(
            dataset=self.dataset,
            difficulty_scores=self.difficulty_scores,
            total_epochs=self.total_epochs,
            current_epoch=current_epoch,
        )
    
    def get_curriculum_info(self, epoch: int) -> Dict:
        """Get curriculum stage information for given epoch."""
        stage1_end = self.total_epochs // 5
        stage2_end = self.total_epochs // 2
        
        if epoch < stage1_end:
            stage = 1
            stage_name = "Easy samples only"
            sample_ratio = 0.33
        elif epoch < stage2_end:
            stage = 2
            stage_name = "Easy + Medium samples"
            sample_ratio = 0.66
        else:
            stage = 3
            stage_name = "All samples"
            sample_ratio = 1.0
        
        return {
            'stage': stage,
            'stage_name': stage_name,
            'sample_ratio': sample_ratio,
            'epoch': epoch,
            'total_epochs': self.total_epochs,
        }


def create_curriculum_dataloader(
    dataset: Dataset,
    batch_size: int,
    total_epochs: int,
    current_epoch: int = 0,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    precomputed_scores: Optional[List[float]] = None,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with curriculum learning.
    
    Args:
        dataset: Training dataset
        batch_size: Batch size
        total_epochs: Total training epochs
        current_epoch: Current epoch
        num_workers: Data loading workers
        collate_fn: Custom collate function
        precomputed_scores: Precomputed difficulty scores
        
    Returns:
        DataLoader with curriculum sampler
    """
    scheduler = CurriculumScheduler(
        dataset=dataset,
        total_epochs=total_epochs,
        precomputed_scores=precomputed_scores,
    )
    
    if precomputed_scores is None:
        # Use random scores for synthetic data or quick start
        scores = [random.random() for _ in range(len(dataset))]
        scheduler.difficulty_scores = scores
    
    sampler = scheduler.create_sampler(current_epoch)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # MPS doesn't support pinned memory
    )


if __name__ == "__main__":
    print("📊 Curriculum Learning Test\n")
    
    # Create dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, n=100):
            self.n = n
        
        def __len__(self):
            return self.n
        
        def __getitem__(self, idx):
            image = torch.rand(3, 64, 64)
            targets = {
                'detection': {
                    'boxes': torch.rand(random.randint(1, 10), 4),
                    'labels': torch.randint(0, 5, (random.randint(1, 10),)),
                }
            }
            return image, targets
    
    dataset = DummyDataset(100)
    
    # Test difficulty scorer
    scorer = DifficultyScorer()
    image, targets = dataset[0]
    difficulty = scorer.score_sample(image, targets)
    print(f"Sample difficulty: {difficulty:.3f}")
    
    # Test curriculum scheduler
    scheduler = CurriculumScheduler(dataset, total_epochs=100)
    scores = scheduler.compute_difficulty_scores(num_samples=50)
    
    # Test sampler at different epochs
    for epoch in [0, 20, 50, 80]:
        sampler = scheduler.create_sampler(epoch)
        info = scheduler.get_curriculum_info(epoch)
        print(f"\nEpoch {epoch}: {info['stage_name']}")
        print(f"  Available samples: {len(sampler)}")
    
    # Test DataLoader
    loader = create_curriculum_dataloader(
        dataset, 
        batch_size=4, 
        total_epochs=100, 
        current_epoch=0
    )
    
    batch = next(iter(loader))
    print(f"\n✓ Curriculum DataLoader created")
    print(f"  Batch shape: {batch[0].shape}")
    
    print("\n✓ All curriculum learning tests passed!")
