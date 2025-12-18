"""
OCR Decoder - Algorithm 6.

Implements license plate text recognition with:
- 20 OCR queries (for character positions)
- Cross-attention to plate features
- CTC-based character sequence decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .base_decoder import BaseDecoder

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig


class CTCHead(nn.Module):
    """
    CTC (Connectionist Temporal Classification) head for character recognition.
    
    Outputs character logits at each position for CTC decoding.
    """
    
    def __init__(self, hidden_dim: int, alphabet: str, max_length: int = 10):
        super().__init__()
        
        self.alphabet = alphabet
        self.num_chars = len(alphabet) + 1  # +1 for blank token
        self.blank_idx = len(alphabet)  # Blank is last
        
        # Character classification at each position
        self.char_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_chars)
        )
        
        # Confidence per character position
        self.conf_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Decoder output [B, num_queries, D]
            
        Returns:
            - 'char_logits': [B, num_queries, num_chars+1] (includes blank)
            - 'char_probs': [B, num_queries, num_chars+1] (softmaxed)
            - 'position_confidence': [B, num_queries, 1]
        """
        char_logits = self.char_proj(x)  # [B, N_q, num_chars]
        char_probs = F.softmax(char_logits, dim=-1)
        position_conf = self.conf_head(x).sigmoid()
        
        return {
            'char_logits': char_logits,
            'char_probs': char_probs,
            'position_confidence': position_conf,
        }
    
    def decode_greedy(self, char_logits: torch.Tensor) -> List[str]:
        """
        Greedy CTC decoding (best path).
        
        Args:
            char_logits: [B, T, num_chars]
            
        Returns:
            List of decoded strings (one per batch)
        """
        # Get best character at each position
        best_chars = char_logits.argmax(dim=-1)  # [B, T]
        
        decoded = []
        for b in range(best_chars.shape[0]):
            chars = best_chars[b].tolist()
            # Remove blanks and collapse repeated
            result = []
            prev = None
            for c in chars:
                if c != self.blank_idx and c != prev:
                    result.append(self.alphabet[c] if c < len(self.alphabet) else '')
                prev = c
            decoded.append(''.join(result))
        
        return decoded


class OCRDecoder(nn.Module):
    """
    OCR Decoder for license plate text recognition.
    
    Uses cross-attention to plate features to focus on exact plate regions.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Base decoder WITH inter-decoder attention (uses plate features)
        self.decoder = BaseDecoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.dim_feedforward,
            num_layers=config.num_decoder_layers,
            dropout=config.dropout,
            has_inter_decoder_attn=True,  # Cross-attend to plate features
            num_queries=config.num_ocr_queries
        )
        
        # CTC output head
        self.head = CTCHead(
            hidden_dim=config.hidden_dim,
            alphabet=config.ocr_alphabet,
            max_length=config.max_plate_length
        )
    
    def forward(
        self,
        memory: torch.Tensor,
        plate_features: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of OCR decoder.
        
        Args:
            memory: Encoder memory [B, N, D]
            plate_features: Features from plate decoder [B, 50, D]
            memory_mask: Optional mask [B, N]
            
        Returns:
            Dictionary with character logits and decoded text
        """
        # Decode with cross-attention to plate features
        ocr_features = self.decoder(
            memory=memory,
            decoder_context=plate_features,
            memory_mask=memory_mask
        )
        
        # Apply CTC head
        outputs = self.head(ocr_features)
        outputs['features'] = ocr_features
        
        return outputs
    
    def decode_plates(
        self,
        char_logits: torch.Tensor,
        plate_confidence: torch.Tensor,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Decode license plate texts with confidence filtering.
        
        Args:
            char_logits: [B, T, num_chars]
            plate_confidence: [B, num_plates, 1] from plate decoder
            threshold: Minimum confidence threshold
            
        Returns:
            List of dictionaries with 'text' and 'confidence'
        """
        decoded_texts = self.head.decode_greedy(char_logits)
        
        results = []
        for i, text in enumerate(decoded_texts):
            # Get confidence (average of position confidences)
            conf = plate_confidence[i].mean().item() if plate_confidence is not None else 1.0
            results.append({
                'text': text,
                'confidence': conf,
                'valid': conf > threshold and len(text) >= 3
            })
        
        return results


if __name__ == "__main__":
    from config import ModelConfig
    
    config = ModelConfig()
    decoder = OCRDecoder(config)
    
    # Dummy inputs
    memory = torch.randn(2, 5376, config.hidden_dim)
    plate_features = torch.randn(2, 50, config.hidden_dim)
    
    # Forward pass
    outputs = decoder(memory, plate_features)
    
    print(f"📊 OCR Decoder Output:")
    print(f"   Features: {outputs['features'].shape}")
    print(f"   Char logits: {outputs['char_logits'].shape}")
    print(f"   Char probs: {outputs['char_probs'].shape}")
    print(f"   Position conf: {outputs['position_confidence'].shape}")
    
    # Test decoding
    decoded = decoder.head.decode_greedy(outputs['char_logits'])
    print(f"   Decoded (random): {decoded}")
    print(f"   Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
