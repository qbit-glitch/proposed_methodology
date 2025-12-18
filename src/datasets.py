"""
Real Dataset Loaders for Multi-Task Training.

Implements dataset loaders for:
- COCO 2017 (Detection)
- Cityscapes (Segmentation)
- CCPD (Plate Detection + OCR)
- MOT17 (Tracking)
"""

import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None
    print("Warning: pycocotools not installed. COCO dataset loading disabled.")


# =============================================================================
# COCO Detection Dataset
# =============================================================================

# COCO vehicle class IDs
COCO_VEHICLE_CLASSES = {
    'car': 3,
    'motorcycle': 4,
    'bus': 6,
    'truck': 8,
}

# Map to our 6 vehicle classes
COCO_TO_MODEL_CLASS = {
    3: 0,   # car -> 0
    4: 1,   # motorcycle -> 1
    6: 2,   # bus -> 2
    8: 3,   # truck -> 3
}


class COCODetectionDataset(Dataset):
    """
    COCO dataset for vehicle detection.
    
    Only loads images containing vehicles (car, motorcycle, bus, truck).
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[callable] = None,
        max_objects: int = 100,
        subset_ratio: float = 1.0,
    ):
        """
        Args:
            root: Path to COCO dataset (containing train2017, val2017, annotations)
            split: 'train' or 'val'
            image_size: Target image size (H, W)
            transform: Optional image transform
            max_objects: Maximum number of objects per image
            subset_ratio: Fraction of dataset to use (0.0-1.0)
        """
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.max_objects = max_objects
        self.subset_ratio = subset_ratio
        
        # Setup paths
        self.image_dir = self.root / f'{split}2017'
        ann_file = self.root / 'annotations' / f'instances_{split}2017.json'
        
        if COCO is None:
            raise ImportError("pycocotools required: pip install pycocotools")
        
        # Load COCO annotations
        print(f"Loading COCO {split} annotations...")
        self.coco = COCO(ann_file)
        
        # Get image IDs containing vehicles
        vehicle_cat_ids = list(COCO_VEHICLE_CLASSES.values())
        self.image_ids = []
        
        for cat_id in vehicle_cat_ids:
            self.image_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
        
        # Remove duplicates
        self.image_ids = list(set(self.image_ids))
        
        # Apply subset ratio
        if subset_ratio < 1.0:
            random.seed(42)  # Reproducible
            n_samples = max(1, int(len(self.image_ids) * subset_ratio))
            self.image_ids = random.sample(self.image_ids, n_samples)
        
        print(f"Found {len(self.image_ids)} images with vehicles (subset_ratio={subset_ratio})")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Resize
        image = image.resize(self.image_size[::-1], Image.BILINEAR)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=list(COCO_VEHICLE_CLASSES.values()))
        anns = self.coco.loadAnns(ann_ids)
        
        # Parse boxes and labels
        boxes = []
        labels = []
        
        for ann in anns[:self.max_objects]:
            if ann['category_id'] not in COCO_TO_MODEL_CLASS:
                continue
            
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Normalize to [0, 1] and convert to cx, cy, w, h
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            
            boxes.append([cx, cy, nw, nh])
            labels.append(COCO_TO_MODEL_CLASS[ann['category_id']])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros(0, 4, dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        
        # Image to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        targets = {
            'detection': {
                'boxes': boxes,
                'labels': labels,
            }
        }
        
        return image, targets


# =============================================================================
# Cityscapes Segmentation Dataset
# =============================================================================

# Cityscapes class mapping to our 3-class problem
# 0: Road/drivable area, 1: Parking/sidewalk, 2: Other
CITYSCAPES_CLASS_MAP = {
    0: 2,   # unlabeled -> other
    1: 2,   # ego vehicle -> other
    2: 2,   # rectification border -> other
    3: 2,   # out of roi -> other
    4: 2,   # static -> other
    5: 2,   # dynamic -> other
    6: 2,   # ground -> other
    7: 0,   # road -> road
    8: 1,   # sidewalk -> parking area
    9: 1,   # parking -> parking area
    10: 2,  # rail track -> other
    11: 2,  # building -> other
    12: 2,  # wall -> other
    13: 2,  # fence -> other
    14: 2,  # guard rail -> other
    15: 2,  # bridge -> other
    16: 2,  # tunnel -> other
    17: 2,  # pole -> other
    18: 2,  # polegroup -> other
    19: 2,  # traffic light -> other
    20: 2,  # traffic sign -> other
    21: 2,  # vegetation -> other
    22: 2,  # terrain -> other
    23: 2,  # sky -> other
    24: 2,  # person -> other
    25: 2,  # rider -> other
    26: 2,  # car -> other (handled by detection)
    27: 2,  # truck -> other
    28: 2,  # bus -> other
    29: 2,  # caravan -> other
    30: 2,  # trailer -> other
    31: 2,  # train -> other
    32: 2,  # motorcycle -> other
    33: 2,  # bicycle -> other
}


class CityscapesDataset(Dataset):
    """
    Cityscapes dataset for scene segmentation.
    
    Maps to 3-class problem: road, parking/sidewalk, other.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[callable] = None,
        subset_ratio: float = 1.0,
    ):
        """
        Args:
            root: Path to Cityscapes dataset
            split: 'train', 'val', or 'test'
            image_size: Target image size (H, W)
            transform: Optional transform
            subset_ratio: Fraction of dataset to use (0.0-1.0)
        """
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.subset_ratio = subset_ratio
        
        # Setup paths - check for different directory structures
        # Standard Cityscapes: leftImg8bit/split/city/*.png
        # Alternative: split/img/*.png
        self.images = []
        self.labels = []
        
        # Try standard structure first
        standard_img_dir = self.root / 'leftImg8bit' / split
        standard_lbl_dir = self.root / 'gtFine' / split
        
        # Try alternative structure
        alt_img_dir = self.root / split / 'img'
        alt_lbl_dir = self.root / split / 'label'
        
        if standard_img_dir.exists():
            # Standard Cityscapes structure
            self.image_dir = standard_img_dir
            self.label_dir = standard_lbl_dir
            
            for city in self.image_dir.iterdir():
                if not city.is_dir():
                    continue
                for img_file in city.glob('*.png'):
                    # Corresponding label file
                    label_name = img_file.stem.replace('_leftImg8bit', '_gtFine_labelIds') + '.png'
                    label_file = self.label_dir / city.name / label_name
                    
                    if label_file.exists():
                        self.images.append(img_file)
                        self.labels.append(label_file)
        
        elif alt_img_dir.exists():
            # Alternative structure: split/img/*.png and split/label/*.png
            self.image_dir = alt_img_dir
            self.label_dir = alt_lbl_dir
            
            for img_file in sorted(self.image_dir.glob('*.png')):
                # Try to find matching label file
                label_file = self.label_dir / img_file.name.replace('_leftImg8bit', '_gtFine_labelIds')
                if not label_file.exists():
                    # Try same name
                    label_file = self.label_dir / img_file.name
                if not label_file.exists():
                    # Try with different naming
                    label_name = img_file.stem.replace('leftImg8bit', 'gtFine_labelIds') + '.png'
                    label_file = self.label_dir / label_name
                    
                if label_file.exists():
                    self.images.append(img_file)
                    self.labels.append(label_file)
                else:
                    # If no label exists, skip or use image as placeholder
                    pass
        
        # Apply subset ratio
        if subset_ratio < 1.0:
            random.seed(42)
            n_samples = max(1, int(len(self.images) * subset_ratio))
            indices = random.sample(range(len(self.images)), n_samples)
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        
        print(f"Found {len(self.images)} Cityscapes {split} images (subset_ratio={subset_ratio})")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        image = image.resize(self.image_size[::-1], Image.BILINEAR)
        
        # Load label
        label_img = Image.open(self.labels[idx])
        label_img = label_img.resize(self.image_size[::-1], Image.NEAREST)
        label = np.array(label_img)
        
        # Handle different label formats (RGB vs grayscale class indices)
        if label.ndim == 3:
            # RGB color-encoded label - convert to grayscale (take first channel or use luminance)
            # For Cityscapes trainIds encoded as RGB: red channel often contains class info
            # Or try computing a simple hash from RGB
            if label.shape[2] == 3:
                # Simple approach: use weighted sum similar to grayscale conversion
                # Map to 3 classes based on color patterns
                mapped_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                
                # Road-like colors (often gray/purple in cityscapes color coding)
                # Sidewalk/parking (often pink/magenta)
                # We'll map based on red channel ranges as a simple heuristic
                r_channel = label[:, :, 0]
                g_channel = label[:, :, 1]
                b_channel = label[:, :, 2]
                
                # Road: typically purple/gray (R=128, G=64, B=128)
                road_mask = (r_channel > 100) & (r_channel < 150) & (b_channel > 100)
                # Sidewalk/parking: typically pink (R=244, G=35, B=232) or (R=250, G=170, B=160)
                parking_mask = (r_channel > 200) & (g_channel < 100)
                
                mapped_label[road_mask] = 0  # Road
                mapped_label[parking_mask] = 1  # Parking/sidewalk
                # Everything else is 2 (other)
                mapped_label[~(road_mask | parking_mask)] = 2
                
                label = mapped_label
            else:
                label = label[:, :, 0]  # Take first channel
        
        # If label is already grayscale with class indices, map to our 3-class problem
        if label.max() > 2:
            mapped_label = np.zeros_like(label)
            for orig_class, new_class in CITYSCAPES_CLASS_MAP.items():
                mapped_label[label == orig_class] = new_class
            label = mapped_label
        
        # Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()
        
        if self.transform:
            image = self.transform(image)
        
        targets = {
            'segmentation': label,
        }
        
        return image, targets


# =============================================================================
# CCPD Dataset (License Plates)
# =============================================================================

# =============================================================================
# License Plate Detection Dataset (YOLO Format)
# =============================================================================

class LicensePlateDataset(Dataset):
    """
    License plate dataset in YOLO format.
    
    Expected structure:
        root/
            images/
                train/
                val/
            labels/
                train/
                val/
    
    Label format (YOLO): class cx cy w h (normalized 0-1)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Any] = None,
        subset_ratio: float = 1.0,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        
        # Find images
        img_dir = self.root / 'images' / split
        label_dir = self.root / 'labels' / split
        
        if not img_dir.exists():
            print(f"Warning: {img_dir} does not exist")
            self.samples = []
            return
        
        # Gather image-label pairs
        self.samples = []
        for img_path in sorted(img_dir.glob('*.jpg')):
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.samples.append((img_path, label_path))
        
        # Also check for png images
        for img_path in sorted(img_dir.glob('*.png')):
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.samples.append((img_path, label_path))
        
        # Apply subset ratio
        if subset_ratio < 1.0:
            random.seed(42)
            n_samples = max(1, int(len(self.samples) * subset_ratio))
            self.samples = random.sample(self.samples, n_samples)
        
        print(f"Found {len(self.samples)} LicensePlate {split} samples (subset_ratio={subset_ratio})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_path, label_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Load YOLO labels: class cx cy w h (normalized)
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    boxes.append([cx, cy, w, h])
                    labels.append(cls)
        
        # Resize image
        image = image.resize(self.image_size[::-1], Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        # Create targets (plate detection is binary: class 0 = plate)
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros(0, 4, dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.long)
        
        targets = {
            'plate': {
                'boxes': boxes_tensor,
                'labels': labels_tensor,
            },
        }
        
        return image, targets


# =============================================================================
# CCPD Dataset (Chinese License Plates)
# =============================================================================

# Chinese provinces for CCPD
CCPD_PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
]

# Characters for CCPD
CCPD_CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9"
]


class CCPDDataset(Dataset):
    """
    CCPD dataset for Chinese license plate detection and recognition.
    
    CCPD filenames encode all annotations:
    - Bounding box vertices
    - License plate text
    - Blur/tilt/distance labels
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[callable] = None,
        return_crops: bool = False,
        subset_ratio: float = 1.0,
    ):
        """
        Args:
            root: Path to CCPD dataset
            split: 'train' or 'val' (80/20 split)
            image_size: Target image size
            transform: Optional transform
            return_crops: If True, also return cropped plate images for OCR
            subset_ratio: Fraction of dataset to use (0.0-1.0)
        """
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform
        self.return_crops = return_crops
        self.subset_ratio = subset_ratio
        
        # Find all images (CCPD uses folder structure)
        all_images = []
        for folder in ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 
                       'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather', 
                       'ccpd_green', 'ccpd_np']:
            folder_path = self.root / folder
            if folder_path.exists():
                all_images.extend(list(folder_path.glob('*.jpg')))
        
        # Simple train/val split
        random.seed(42)
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.8)
        
        if split == 'train':
            self.images = all_images[:split_idx]
        else:
            self.images = all_images[split_idx:]
        
        # Apply subset ratio
        if subset_ratio < 1.0:
            n_samples = max(1, int(len(self.images) * subset_ratio))
            self.images = self.images[:n_samples]
        
        print(f"Found {len(self.images)} CCPD {split} images (subset_ratio={subset_ratio})")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def _parse_filename(self, filename: str) -> Dict:
        """
        Parse CCPD filename to extract annotations.
        
        CCPD filename format: ID-AREA-BBOX-VERTICES-PLATE_INDICES-BRIGHTNESS-BLUR.jpg
        Example: 0139463601533-90_79-218,465_448,528-460,534_233,527_206,452_433,459-0_0_0_19_26_25_25-129-17
        - parts[0]: unique ID
        - parts[1]: area (not used)
        - parts[2]: bounding box corners (2 corners)
        - parts[3]: 4 vertices (x1,y1_x2,y2_x3,y3_x4,y4)
        - parts[4]: plate indices (province_c1_c2_c3_c4_c5_c6), 7 chars total
        - parts[5]: brightness
        - parts[6]: blur
        """
        parts = filename.split('-')
        
        if len(parts) < 7:
            return None
        
        try:
            # Parse 4 vertices from parts[3] (comma separates x,y)
            vertices_str = parts[3]  # e.g., '460,534_233,527_206,452_433,459'
            vertices = []
            for v in vertices_str.split('_'):
                coords = v.split(',')
                if len(coords) == 2:
                    vertices.append((int(coords[0]), int(coords[1])))
            
            # Parse plate text from parts[4] (province + 6 characters)
            if len(parts) >= 5:
                plate_indices = parts[4].split('_')
                if len(plate_indices) >= 1:
                    # First index is province
                    province_idx = int(plate_indices[0])
                    plate_text = CCPD_PROVINCES[province_idx] if province_idx < len(CCPD_PROVINCES) else ""
                    # Remaining indices are characters
                    for idx_str in plate_indices[1:]:
                        idx = int(idx_str)
                        if idx < len(CCPD_CHARS):
                            plate_text += CCPD_CHARS[idx]
            else:
                plate_text = ""
            
            return {
                'vertices': vertices,
                'plate_text': plate_text,
            }
        except Exception as e:
            return None
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_path = self.images[idx]
        
        # Parse annotation from filename
        ann = self._parse_filename(img_path.stem)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        if ann and len(ann['vertices']) >= 4:
            # Get bounding box from vertices
            xs = [v[0] for v in ann['vertices']]
            ys = [v[1] for v in ann['vertices']]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            
            # Normalize
            cx = ((x1 + x2) / 2) / orig_w
            cy = ((y1 + y2) / 2) / orig_h
            w = (x2 - x1) / orig_w
            h = (y2 - y1) / orig_h
            
            boxes = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
            plate_text = [ann['plate_text']]
        else:
            boxes = torch.zeros(0, 4, dtype=torch.float32)
            plate_text = []
        
        # Resize image
        image = image.resize(self.image_size[::-1], Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        targets = {
            'plate': {
                'boxes': boxes,
                'labels': torch.zeros(len(boxes), dtype=torch.long),  # All plates = class 0
            },
            'ocr': plate_text,
        }
        
        return image, targets


# =============================================================================
# MOT17 Tracking Dataset
# =============================================================================

class MOT17Dataset(Dataset):
    """
    MOT17 dataset for multi-object tracking.
    
    Returns pairs of consecutive frames with tracking annotations.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[callable] = None,
        frame_gap: int = 1,
        subset_ratio: float = 1.0,
    ):
        """
        Args:
            root: Path to MOT17 dataset
            split: 'train' or 'test'
            image_size: Target image size
            transform: Optional transform
            frame_gap: Gap between frame pairs (default: 1)
            subset_ratio: Fraction of dataset to use (0.0-1.0)
        """
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.frame_gap = frame_gap
        self.subset_ratio = subset_ratio
        
        # Load all sequences
        self.samples = []  # (seq_path, frame1_idx, frame2_idx)
        
        seq_dir = self.root / split
        for seq in seq_dir.iterdir():
            if not seq.is_dir():
                continue
            if not seq.name.startswith('MOT17-'):
                continue
            
            img_dir = seq / 'img1'
            gt_file = seq / 'gt' / 'gt.txt'
            
            if not img_dir.exists():
                continue
            
            # Count frames
            frames = sorted(img_dir.glob('*.jpg'))
            num_frames = len(frames)
            
            # Load ground truth
            gt_data = {}
            if gt_file.exists():
                with open(gt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            frame_id = int(parts[0])
                            track_id = int(parts[1])
                            x, y, w, h = map(float, parts[2:6])
                            
                            if frame_id not in gt_data:
                                gt_data[frame_id] = []
                            gt_data[frame_id].append({
                                'track_id': track_id,
                                'bbox': [x, y, w, h]  # x, y, w, h
                            })
            
            # Create frame pairs
            for i in range(1, num_frames - self.frame_gap):
                self.samples.append((seq, i, i + self.frame_gap, gt_data))
        
        # Apply subset ratio
        if subset_ratio < 1.0:
            random.seed(42)
            n_samples = max(1, int(len(self.samples) * subset_ratio))
            self.samples = random.sample(self.samples, n_samples)
        
        print(f"Found {len(self.samples)} MOT17 {split} frame pairs (subset_ratio={subset_ratio})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        seq_path, frame1_idx, frame2_idx, gt_data = self.samples[idx]
        
        # Load images
        img1_path = seq_path / 'img1' / f'{frame1_idx:06d}.jpg'
        img2_path = seq_path / 'img1' / f'{frame2_idx:06d}.jpg'
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        orig_w, orig_h = img1.size
        
        # Resize
        img1 = img1.resize(self.image_size[::-1], Image.BILINEAR)
        img2 = img2.resize(self.image_size[::-1], Image.BILINEAR)
        
        # Get annotations for both frames
        gt1 = gt_data.get(frame1_idx, [])
        gt2 = gt_data.get(frame2_idx, [])
        
        # Build tracking targets
        # For now, we use the current frame detections and previous boxes
        boxes1, boxes2 = [], []
        track_ids1, track_ids2 = [], []
        
        for ann in gt1:
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes1.append([cx, cy, nw, nh])
            track_ids1.append(ann['track_id'])
        
        for ann in gt2:
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes2.append([cx, cy, nw, nh])
            track_ids2.append(ann['track_id'])
        
        # Convert to tensors
        img2_tensor = torch.from_numpy(np.array(img2)).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            img2_tensor = self.transform(img2_tensor)
        
        # Build association matrix (which track in frame1 matches which in frame2)
        num_queries = 100
        associations = torch.zeros(num_queries, num_queries, dtype=torch.long)
        trajectory = torch.zeros(num_queries, 4)
        
        # Match by track ID
        for i, tid1 in enumerate(track_ids1[:num_queries]):
            for j, tid2 in enumerate(track_ids2[:num_queries]):
                if tid1 == tid2:
                    associations[i, j] = 1
                    if i < len(boxes1) and j < len(boxes2):
                        trajectory[i] = torch.tensor(boxes2[j]) - torch.tensor(boxes1[i])
        
        # Pad track IDs
        padded_track_ids = torch.full((num_queries,), -1, dtype=torch.long)
        for i, tid in enumerate(track_ids2[:num_queries]):
            padded_track_ids[i] = tid
        
        targets = {
            'detection': {
                'boxes': torch.tensor(boxes2, dtype=torch.float32) if boxes2 else torch.zeros(0, 4),
                'labels': torch.zeros(len(boxes2), dtype=torch.long),
            },
            'tracking': {
                'track_ids': padded_track_ids,
                'associations': associations,
                'trajectory': trajectory,
                'prev_boxes': torch.tensor(boxes1, dtype=torch.float32) if boxes1 else torch.zeros(0, 4),
            },
        }
        
        return img2_tensor, targets


# =============================================================================
# Multi-Task Combined Dataset
# =============================================================================

class MultiTaskDataset(Dataset):
    """
    Combined dataset that merges all task-specific datasets.
    
    Randomly samples from each dataset and combines annotations.
    """
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        primary_task: str = 'detection',
    ):
        """
        Args:
            datasets: Dict of task_name -> Dataset
            primary_task: Primary task to determine dataset length
        """
        self.datasets = datasets
        self.primary_task = primary_task
        self.primary_dataset = datasets[primary_task]
    
    def __len__(self) -> int:
        return len(self.primary_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Get primary sample
        image, targets = self.primary_dataset[idx]
        
        # Randomly sample from other datasets to fill missing targets
        for task_name, dataset in self.datasets.items():
            if task_name == self.primary_task:
                continue
            
            if task_name not in targets:
                rand_idx = random.randint(0, len(dataset) - 1)
                _, other_targets = dataset[rand_idx]
                targets.update(other_targets)
        
        return image, targets


# =============================================================================
# Collate Functions
# =============================================================================

def multi_task_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, Dict]:
    """Collate function for multi-task training."""
    images = torch.stack([item[0] for item in batch])
    
    targets = {}
    
    # Detection
    if any('detection' in item[1] for item in batch):
        targets['detection'] = [item[1].get('detection', {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}) for item in batch]
    
    # Segmentation
    if any('segmentation' in item[1] for item in batch):
        seg_masks = []
        for item in batch:
            if 'segmentation' in item[1]:
                seg_masks.append(item[1]['segmentation'])
            else:
                seg_masks.append(torch.zeros(images.size(2), images.size(3), dtype=torch.long))
        targets['segmentation'] = torch.stack(seg_masks)
    
    # Plate
    if any('plate' in item[1] for item in batch):
        targets['plate'] = [item[1].get('plate', {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}) for item in batch]
    
    # OCR
    if any('ocr' in item[1] for item in batch):
        targets['ocr'] = []
        for item in batch:
            if 'ocr' in item[1]:
                targets['ocr'].extend(item[1]['ocr'])
    
    # Tracking
    if any('tracking' in item[1] for item in batch):
        targets['tracking'] = [item[1].get('tracking', {}) for item in batch]
    
    return images, targets


# =============================================================================
# Factory Functions
# =============================================================================

def create_coco_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (512, 512),
    subset_ratio: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """Create COCO data loaders for detection."""
    train_dataset = COCODetectionDataset(data_dir, 'train', image_size, subset_ratio=subset_ratio)
    val_dataset = COCODetectionDataset(data_dir, 'val', image_size, subset_ratio=subset_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=multi_task_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=multi_task_collate_fn, num_workers=num_workers)
    
    return train_loader, val_loader


def create_cityscapes_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (512, 512),
    subset_ratio: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """Create Cityscapes data loaders for segmentation."""
    train_dataset = CityscapesDataset(data_dir, 'train', image_size, subset_ratio=subset_ratio)
    val_dataset = CityscapesDataset(data_dir, 'val', image_size, subset_ratio=subset_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=multi_task_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=multi_task_collate_fn, num_workers=num_workers)
    
    return train_loader, val_loader


def create_ccpd_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (512, 512),
    subset_ratio: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """Create CCPD data loaders for plate detection and OCR."""
    train_dataset = CCPDDataset(data_dir, 'train', image_size, subset_ratio=subset_ratio)
    val_dataset = CCPDDataset(data_dir, 'val', image_size, subset_ratio=subset_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=multi_task_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=multi_task_collate_fn, num_workers=num_workers)
    
    return train_loader, val_loader


def create_mot17_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (512, 512),
    subset_ratio: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """Create MOT17 data loaders for tracking."""
    train_dataset = MOT17Dataset(data_dir, 'train', image_size, subset_ratio=subset_ratio)
    # MOT17 test set has no labels, so use train for both
    val_dataset = MOT17Dataset(data_dir, 'train', image_size, subset_ratio=subset_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=multi_task_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=multi_task_collate_fn, num_workers=num_workers)
    
    return train_loader, val_loader


def create_license_plate_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (512, 512),
    subset_ratio: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """Create License Plate data loaders for plate detection (YOLO format)."""
    train_dataset = LicensePlateDataset(data_dir, 'train', image_size, subset_ratio=subset_ratio)
    val_dataset = LicensePlateDataset(data_dir, 'val', image_size, subset_ratio=subset_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=multi_task_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=multi_task_collate_fn, num_workers=num_workers)
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Dataset Loaders Test")
    print("=" * 50)
    
    # Test synthetic fallback
    print("\n✓ Dataset loaders module loaded successfully")
    print("\nAvailable datasets:")
    print("  - COCODetectionDataset (requires pycocotools)")
    print("  - CityscapesDataset")
    print("  - CCPDDataset")
    print("  - MOT17Dataset")
    print("  - MultiTaskDataset (combines all)")
