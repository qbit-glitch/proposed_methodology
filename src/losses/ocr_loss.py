"""
OCR Loss Functions.

Implements CTC (Connectionist Temporal Classification) Loss for sequence prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class CTCLoss(nn.Module):
    """
    CTC Loss for OCR sequence prediction.
    
    Handles variable-length text without requiring character-level alignment.
    """
    
    def __init__(self, blank_idx: int = 36, reduction: str = 'mean'):
        super().__init__()
        self.blank_idx = blank_idx
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[torch.Tensor],  # List of [seq_len] label tensors
        input_lengths: torch.Tensor = None,
        target_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CTC loss.
        
        Args:
            outputs:
                - 'char_logits': [B, T, num_chars] logits
            targets: List of label tensors for each sample
            input_lengths: [B] length of each input sequence
            target_lengths: [B] length of each target sequence
            
        Returns:
            Dict with 'loss_ocr_ctc'
        """
        logits = outputs['char_logits']  # [B, T, C]
        B, T, C = logits.shape
        original_device = logits.device
        
        # CTC loss not supported on MPS - use CPU fallback
        use_cpu_fallback = original_device.type == 'mps'
        if use_cpu_fallback:
            logits = logits.cpu()
        
        # CTC expects [T, B, C]
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, C]
        
        # Prepare targets
        if isinstance(targets, list):
            # Concatenate all targets
            targets_cat = torch.cat(targets) if len(targets) > 0 and targets[0].numel() > 0 else torch.tensor([], dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        else:
            targets_cat = targets
        
        # Move to same device as logits (CPU if fallback)
        compute_device = logits.device
        targets_cat = targets_cat.to(compute_device)
        if target_lengths is not None:
            target_lengths = target_lengths.to(compute_device)
        
        # Input lengths (all positions used)
        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=compute_device)
        else:
            input_lengths = input_lengths.to(compute_device)
        
        # Handle empty targets
        if targets_cat.numel() == 0:
            return {'loss_ocr_ctc': torch.tensor(0.0, device=original_device)}
        
        # Compute CTC loss (on CPU if MPS)
        loss = self.ctc_loss(
            log_probs,
            targets_cat,
            input_lengths,
            target_lengths
        )
        
        # Move result back to original device
        if use_cpu_fallback:
            loss = loss.to(original_device)
        
        return {'loss_ocr_ctc': loss}


class OCRLoss(nn.Module):
    """
    Combined OCR loss with CTC and optional auxiliary losses.
    """
    
    def __init__(
        self,
        alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        ctc_weight: float = 1.0,
    ):
        super().__init__()
        self.alphabet = alphabet
        self.blank_idx = len(alphabet)  # Blank token at end
        self.ctc_loss = CTCLoss(blank_idx=self.blank_idx)
        self.ctc_weight = ctc_weight
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text string to label tensor."""
        labels = []
        for char in text.upper():
            if char in self.alphabet:
                labels.append(self.alphabet.index(char))
        return torch.tensor(labels, dtype=torch.long)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[str],  # List of text strings
    ) -> Dict[str, torch.Tensor]:
        """
        Compute OCR loss.
        
        Args:
            outputs:
                - 'char_logits': [B, T, num_chars]
            targets: List of text strings (should match batch size)
            
        Returns:
            Dict with 'loss_ocr_ctc'
        """
        logits = outputs['char_logits']  # [B, T, C]
        B = logits.shape[0]
        device = logits.device
        
        # Handle case where targets don't match batch size
        if len(targets) != B:
            # Take first B targets or pad with empty
            if len(targets) > B:
                targets = targets[:B]
            else:
                targets = targets + [''] * (B - len(targets))
        
        # Encode text targets
        encoded_targets = [self.encode_text(t) for t in targets]
        
        # Filter out empty targets
        valid_indices = [i for i, t in enumerate(encoded_targets) if t.numel() > 0]
        
        if len(valid_indices) == 0:
            return {'loss_ocr_ctc': torch.tensor(0.0, device=device)}
        
        # Compute CTC loss only for valid samples
        valid_logits = logits[valid_indices]  # [B', T, C]
        valid_targets = [encoded_targets[i] for i in valid_indices]
        
        outputs_valid = {'char_logits': valid_logits}
        losses = self.ctc_loss(outputs_valid, valid_targets)
        losses['loss_ocr_ctc'] = self.ctc_weight * losses['loss_ocr_ctc']
        
        return losses


if __name__ == "__main__":
    # Test OCR loss
    loss_fn = OCRLoss()
    
    # Dummy outputs and targets
    outputs = {
        'char_logits': torch.randn(2, 20, 37),  # 36 chars + blank
    }
    targets = ["DL01AB1234", "MH12XY5678"]
    
    losses = loss_fn(outputs, targets)
    
    print("📊 OCR Loss Test:")
    for name, value in losses.items():
        print(f"   {name}: {value.item():.4f}")
