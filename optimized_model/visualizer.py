"""
Visualizer for Unified Multi-Task Transformer outputs.

Provides visualization functions for:
- Detection bounding boxes with class labels
- Segmentation masks overlay
- License plate detection
- OCR text display
- Tracking trajectories
- Alert highlighting (red boxes for violations)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import os


# Color palette for different classes
VEHICLE_COLORS = {
    'car': (0, 255, 0),       # Green
    'scooty': (255, 165, 0),   # Orange
    'bike': (255, 255, 0),     # Yellow
    'bus': (0, 0, 255),        # Blue
    'truck': (128, 0, 128),    # Purple
    'auto': (0, 255, 255),     # Cyan
    'background': (128, 128, 128),
}

SEGMENTATION_COLORS = {
    'background': (0, 0, 0, 0),       # Transparent
    'driveway': (100, 100, 255, 128),  # Blue with alpha
    'footpath': (100, 255, 100, 128),  # Green with alpha
}

ALERT_COLOR = (255, 0, 0)  # Red for violations


class Visualizer:
    """Visualization utilities for model outputs."""
    
    def __init__(
        self,
        vehicle_classes: List[str] = None,
        seg_classes: List[str] = None,
        font_size: int = 12,
    ):
        self.vehicle_classes = vehicle_classes or [
            "car", "scooty", "bike", "bus", "truck", "auto"
        ]
        self.seg_classes = seg_classes or ["background", "driveway", "footpath"]
        self.font_size = font_size
        
        # Try to load a font, fall back to default
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            self.font = ImageFont.load_default()
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor [C, H, W] or [H, W, C] to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first batch
        
        if tensor.shape[0] in [1, 3]:  # [C, H, W]
            tensor = tensor.permute(1, 2, 0)
        
        # Denormalize if needed
        if tensor.min() < 0:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        # Convert to uint8
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if array.shape[2] == 1:
            array = array.squeeze(2)
        
        return Image.fromarray(array)
    
    def draw_bboxes(
        self,
        image: Image.Image,
        bboxes: torch.Tensor,  # [N, 4] normalized
        class_logits: torch.Tensor,  # [N, num_classes]
        scores_threshold: float = 0.5,
        alerts: Optional[List] = None,
    ) -> Image.Image:
        """
        Draw bounding boxes with class labels on image.
        
        Args:
            image: PIL Image
            bboxes: [N, 4] normalized bboxes (x_center, y_center, w, h)
            class_logits: [N, num_classes+1] class logits
            scores_threshold: Minimum confidence threshold
            alerts: Optional list of Alert objects for violation highlighting
            
        Returns:
            Image with drawn bboxes
        """
        draw = ImageDraw.Draw(image)
        W, H = image.size
        
        # Get class probabilities
        class_probs = F.softmax(class_logits, dim=-1)
        
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            probs = class_probs[i]
            
            class_idx = probs.argmax().item()
            score = probs[class_idx].item()
            
            # Skip background or low confidence
            if class_idx >= len(self.vehicle_classes) or score < scores_threshold:
                continue
            
            class_name = self.vehicle_classes[class_idx]
            
            # Convert normalized bbox to pixel coords
            x_center, y_center, w, h = bbox.tolist()
            x1 = int((x_center - w / 2) * W)
            y1 = int((y_center - h / 2) * H)
            x2 = int((x_center + w / 2) * W)
            y2 = int((y_center + h / 2) * H)
            
            # Check if this is a violation
            is_violation = False
            if alerts:
                for alert in alerts:
                    if alert.track_id == i and alert.alert_type == "RED":
                        is_violation = True
                        break
            
            # Choose color
            if is_violation:
                color = ALERT_COLOR
                label = f"{class_name} VIOLATION"
            else:
                color = VEHICLE_COLORS.get(class_name, (255, 255, 255))
                label = f"{class_name} {score:.2f}"
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2 if not is_violation else 4)
            
            # Draw label background
            text_bbox = draw.textbbox((x1, y1 - 15), label, font=self.font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 15), label, fill=(255, 255, 255), font=self.font)
        
        return image
    
    def draw_segmentation(
        self,
        image: Image.Image,
        segmentation_mask: torch.Tensor,  # [H, W] or [num_classes, H, W]
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Overlay segmentation mask on image.
        
        Args:
            image: PIL Image
            segmentation_mask: [H, W] class indices or [num_classes, H, W] probabilities
            alpha: Transparency for overlay
            
        Returns:
            Image with segmentation overlay
        """
        W, H = image.size
        
        # Handle different input formats
        if segmentation_mask.dim() == 3:
            # [num_classes, H, W] -> [H, W]
            segmentation_mask = segmentation_mask.argmax(dim=0)
        
        # Resize mask to image size
        mask = F.interpolate(
            segmentation_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W),
            mode='nearest'
        ).squeeze().long()
        
        # Create overlay
        overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        mask_np = mask.cpu().numpy()
        
        for class_idx, class_name in enumerate(self.seg_classes):
            if class_idx == 0:  # Skip background
                continue
            
            color = SEGMENTATION_COLORS.get(class_name, (128, 128, 128, 128))
            
            # Create mask for this class
            class_mask = (mask_np == class_idx)
            
            # Draw pixels
            for y in range(H):
                for x in range(W):
                    if class_mask[y, x]:
                        overlay.putpixel((x, y), color)
        
        # Blend with original image
        image = image.convert('RGBA')
        result = Image.alpha_composite(image, overlay)
        
        return result.convert('RGB')
    
    def draw_plates(
        self,
        image: Image.Image,
        plate_bboxes: torch.Tensor,  # [N, 4]
        plate_confidences: torch.Tensor,  # [N, 1]
        ocr_texts: Optional[List[str]] = None,
        threshold: float = 0.3,
    ) -> Image.Image:
        """Draw license plate bounding boxes with OCR text."""
        draw = ImageDraw.Draw(image)
        W, H = image.size
        
        for i in range(plate_bboxes.shape[0]):
            conf = plate_confidences[i].item()
            if conf < threshold:
                continue
            
            bbox = plate_bboxes[i]
            x_center, y_center, w, h = bbox.tolist()
            x1 = int((x_center - w / 2) * W)
            y1 = int((y_center - h / 2) * H)
            x2 = int((x_center + w / 2) * W)
            y2 = int((y_center + h / 2) * H)
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
            
            # Draw OCR text if available
            if ocr_texts and i < len(ocr_texts):
                text = ocr_texts[i]
                draw.text((x1, y2 + 2), text, fill=(255, 255, 0), font=self.font)
        
        return image
    
    def draw_tracking(
        self,
        image: Image.Image,
        track_features: torch.Tensor,
        bboxes: torch.Tensor,
        track_ids: List[int],
        trajectories: Optional[Dict[int, List[Tuple]]] = None,
    ) -> Image.Image:
        """Draw tracking information with trajectories."""
        draw = ImageDraw.Draw(image)
        W, H = image.size
        
        # Color palette for track IDs
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]
        
        for i, track_id in enumerate(track_ids):
            color = colors[track_id % len(colors)]
            bbox = bboxes[i]
            
            x_center, y_center, w, h = bbox.tolist()
            x1 = int((x_center - w / 2) * W)
            y1 = int((y_center - h / 2) * H)
            x2 = int((x_center + w / 2) * W)
            y2 = int((y_center + h / 2) * H)
            
            # Draw bbox with track ID
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 12), f"ID:{track_id}", fill=color, font=self.font)
            
            # Draw trajectory if available
            if trajectories and track_id in trajectories:
                traj = trajectories[track_id]
                for j in range(1, len(traj)):
                    p1 = (int(traj[j-1][0] * W), int(traj[j-1][1] * H))
                    p2 = (int(traj[j][0] * W), int(traj[j][1] * H))
                    draw.line([p1, p2], fill=color, width=2)
        
        return image
    
    def visualize_all(
        self,
        image: torch.Tensor,
        outputs: Dict,
        alerts: Optional[List] = None,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Create complete visualization with all outputs.
        
        Args:
            image: Input image tensor [C, H, W]
            outputs: Model outputs dictionary
            alerts: Optional list of Alert objects
            save_path: Optional path to save the result
            
        Returns:
            Combined visualization image
        """
        # Convert tensor to PIL
        pil_image = self.tensor_to_pil(image)
        
        # 1. Draw segmentation first (as background)
        if 'segmentation' in outputs:
            seg_masks = outputs['segmentation']['masks'][0]  # [N_q, H, W]
            seg_logits = outputs['segmentation']['class_logits'][0]  # [N_q, 3]
            
            # Simple argmax for visualization
            combined_mask = seg_masks.mean(dim=0)  # Average across queries
            combined_mask = (combined_mask > 0).long()
            pil_image = self.draw_segmentation(pil_image, combined_mask)
        
        # 2. Draw detection bboxes
        if 'detection' in outputs:
            pil_image = self.draw_bboxes(
                pil_image,
                outputs['detection']['bbox'][0],
                outputs['detection']['class_logits'][0],
                alerts=alerts
            )
        
        # 3. Draw plate detections with OCR
        if 'plate' in outputs:
            ocr_texts = None
            if 'ocr' in outputs:
                # Decode OCR
                from .decoders.ocr_decoder import OCRDecoder
                # Would need decoder instance to properly decode
                pass
            
            pil_image = self.draw_plates(
                pil_image,
                outputs['plate']['plate_bbox'][0],
                outputs['plate']['plate_confidence'][0],
                ocr_texts=ocr_texts
            )
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pil_image.save(save_path)
            print(f"✓ Saved visualization to {save_path}")
        
        return pil_image


def create_visualization_grid(
    images: List[Image.Image],
    titles: List[str],
    cols: int = 2,
) -> Image.Image:
    """Create a grid of images with titles."""
    n = len(images)
    rows = (n + cols - 1) // cols
    
    # Get max dimensions
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    
    title_height = 30
    padding = 10
    
    grid_w = cols * (max_w + padding) + padding
    grid_h = rows * (max_h + title_height + padding) + padding
    
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        
        x = padding + col * (max_w + padding)
        y = padding + row * (max_h + title_height + padding)
        
        # Draw title
        draw.text((x, y), title, fill=(0, 0, 0))
        
        # Paste image
        grid.paste(img, (x, y + title_height))
    
    return grid


if __name__ == "__main__":
    # Test visualizer
    visualizer = Visualizer()
    
    # Create dummy image
    dummy_image = torch.rand(3, 512, 512)
    pil_img = visualizer.tensor_to_pil(dummy_image)
    
    # Create dummy outputs
    N = 10
    dummy_outputs = {
        'detection': {
            'bbox': torch.rand(1, N, 4),
            'class_logits': torch.randn(1, N, 7),
        },
        'segmentation': {
            'masks': torch.rand(1, 50, 64, 64),
            'class_logits': torch.randn(1, 50, 3),
        },
        'plate': {
            'plate_bbox': torch.rand(1, 5, 4),
            'plate_confidence': torch.rand(1, 5, 1),
        }
    }
    
    # Visualize
    result = visualizer.visualize_all(
        dummy_image,
        dummy_outputs,
        save_path="outputs/test_visualization.png"
    )
    
    print(f"✓ Visualization test complete: {result.size}")
