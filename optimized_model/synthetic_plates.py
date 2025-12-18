"""
Synthetic License Plate Generator (Algorithm S2.6).

Generates synthetic Indian license plates for OCR training:
- Format: [A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4} (e.g., "DL01AB1234")
- Realistic distortions (rotation, perspective, blur, noise)
- Multiple background colors (white, yellow)
"""

import random
import math
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw, ImageFont
import io
import os


# Indian state codes
INDIAN_STATE_CODES = [
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'GA',
    'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
    'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK',
    'TN', 'TS', 'TR', 'UK', 'UP', 'WB'
]


def generate_plate_text() -> str:
    """
    Generate random Indian license plate text.
    
    Format: SS00XX0000 (State code, District, Series, Number)
    """
    # State code (2 letters)
    state = random.choice(INDIAN_STATE_CODES)
    
    # District code (2 digits)
    district = f"{random.randint(1, 99):02d}"
    
    # Series (1-2 letters)
    letters = 'ABCDEFGHJKLMNPRSTUVWXYZ'  # Exclude I, O, Q for clarity
    series = random.choice(letters) + random.choice(letters)
    
    # Number (4 digits)
    number = f"{random.randint(1, 9999):04d}"
    
    return f"{state}{district}{series}{number}"


class SyntheticPlateGenerator:
    """
    Generator for synthetic license plate images.
    
    Creates realistic Indian license plates with various augmentations.
    """
    
    def __init__(
        self,
        plate_width: int = 200,
        plate_height: int = 50,
        output_size: Tuple[int, int] = (128, 32),
        # Distortion parameters
        rotation_range: Tuple[float, float] = (-15, 15),
        perspective_range: float = 0.1,
        blur_range: Tuple[float, float] = (0, 2),
        noise_std: float = 0.02,
        # Colors
        bg_colors: List[str] = None,
        text_colors: List[str] = None,
    ):
        self.plate_width = plate_width
        self.plate_height = plate_height
        self.output_size = output_size
        
        self.rotation_range = rotation_range
        self.perspective_range = perspective_range
        self.blur_range = blur_range
        self.noise_std = noise_std
        
        # Default colors
        self.bg_colors = bg_colors or ['white', '#FFFDD0', '#F5F5DC']  # White/cream plates
        self.text_colors = text_colors or ['black', '#1A1A1A', '#2B2B2B']
        
        # Yellow plates for commercial vehicles
        self.commercial_bg = ['#FFD700', '#FFC000', '#FFB347']
        self.commercial_text = ['black', '#1A1A1A']
        
        # Try to load fonts
        self.fonts = self._load_fonts()
    
    def _load_fonts(self) -> List:
        """Load available fonts for plate text."""
        fonts = []
        
        # Try common system fonts that look like plate fonts
        font_names = [
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/HelveticaNeue.ttc',
            '/System/Library/Fonts/Monaco.dfont',
            '/Library/Fonts/Arial.ttf',
            '/Library/Fonts/Arial Bold.ttf',
        ]
        
        for font_path in font_names:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, size=40)
                    fonts.append(font)
                except:
                    pass
        
        # Fallback to default font
        if not fonts:
            try:
                fonts.append(ImageFont.load_default())
            except:
                fonts.append(None)
        
        return fonts
    
    def generate(
        self,
        text: Optional[str] = None,
        commercial: bool = False,
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate a synthetic license plate image.
        
        Args:
            text: Plate text (auto-generated if None)
            commercial: Whether to use commercial vehicle colors
            
        Returns:
            Tuple of (image tensor [3, H, W], text string)
        """
        if text is None:
            text = generate_plate_text()
        
        # Create base plate
        plate = self._create_base_plate(text, commercial)
        
        # Apply distortions
        plate = self._apply_rotation(plate)
        plate = self._apply_perspective(plate)
        plate = self._apply_blur(plate)
        
        # Convert to tensor
        plate_tensor = self._pil_to_tensor(plate)
        
        # Add noise
        plate_tensor = self._add_noise(plate_tensor)
        
        # Resize to output size
        plate_tensor = F.interpolate(
            plate_tensor.unsqueeze(0),
            size=self.output_size[::-1],  # PIL uses (W, H), tensor uses (H, W)
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Clamp values
        plate_tensor = torch.clamp(plate_tensor, 0, 1)
        
        return plate_tensor, text
    
    def _create_base_plate(self, text: str, commercial: bool = False) -> Image.Image:
        """Create the base plate image with text."""
        # Select colors
        if commercial:
            bg_color = random.choice(self.commercial_bg)
            text_color = random.choice(self.commercial_text)
        else:
            bg_color = random.choice(self.bg_colors)
            text_color = random.choice(self.text_colors)
        
        # Create image
        plate = Image.new('RGB', (self.plate_width, self.plate_height), bg_color)
        draw = ImageDraw.Draw(plate)
        
        # Select font
        font = random.choice(self.fonts)
        
        # Add border
        border_width = 2
        draw.rectangle(
            [border_width, border_width, 
             self.plate_width - border_width - 1, self.plate_height - border_width - 1],
            outline='black',
            width=border_width
        )
        
        # Draw text centered
        if font:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width = len(text) * 15
                text_height = 30
        else:
            text_width = len(text) * 15
            text_height = 30
        
        x = (self.plate_width - text_width) // 2
        y = (self.plate_height - text_height) // 2
        
        if font:
            draw.text((x, y), text, fill=text_color, font=font)
        else:
            draw.text((x, y), text, fill=text_color)
        
        return plate
    
    def _apply_rotation(self, image: Image.Image) -> Image.Image:
        """Apply random rotation."""
        angle = random.uniform(*self.rotation_range)
        
        # Expand canvas to avoid cutting
        expanded_size = (
            int(self.plate_width * 1.5),
            int(self.plate_height * 2)
        )
        expanded = Image.new('RGB', expanded_size, 'white')
        
        # Paste centered
        x = (expanded_size[0] - self.plate_width) // 2
        y = (expanded_size[1] - self.plate_height) // 2
        expanded.paste(image, (x, y))
        
        # Rotate
        rotated = expanded.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor='white')
        
        # Crop back to original size (centered)
        left = (expanded_size[0] - self.plate_width) // 2
        top = (expanded_size[1] - self.plate_height) // 2
        cropped = rotated.crop((left, top, left + self.plate_width, top + self.plate_height))
        
        return cropped
    
    def _apply_perspective(self, image: Image.Image) -> Image.Image:
        """Apply random perspective transform."""
        width, height = image.size
        
        # Random perspective distortion
        offset = int(width * self.perspective_range)
        
        # Four corner offsets
        tl = (random.randint(0, offset), random.randint(0, offset))
        tr = (width - random.randint(0, offset), random.randint(0, offset))
        bl = (random.randint(0, offset), height - random.randint(0, offset))
        br = (width - random.randint(0, offset), height - random.randint(0, offset))
        
        # Original corners
        orig = [(0, 0), (width, 0), (width, height), (0, height)]
        # Transformed corners
        new = [tl, tr, br, bl]
        
        # Compute perspective transform coefficients
        coeffs = self._find_perspective_coeffs(new, orig)
        
        # Apply transform
        transformed = image.transform(
            (width, height),
            Image.PERSPECTIVE,
            coeffs,
            Image.BICUBIC,
            fillcolor='white'
        )
        
        return transformed
    
    def _find_perspective_coeffs(self, source_coords, target_coords):
        """Find perspective transform coefficients."""
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        
        A = torch.tensor(matrix, dtype=torch.float32)
        B = torch.tensor([s[0] for s in source_coords] + [s[1] for s in source_coords], dtype=torch.float32)
        
        try:
            res = torch.linalg.lstsq(A, B).solution
            return tuple(res.numpy())
        except:
            return (1, 0, 0, 0, 1, 0, 0, 0)
    
    def _apply_blur(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur."""
        from PIL import ImageFilter
        
        sigma = random.uniform(*self.blur_range)
        if sigma > 0.5:
            radius = int(sigma * 2)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return image
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to torch tensor."""
        import numpy as np
        
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
        
        return tensor
    
    def _add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        noise = torch.randn_like(tensor) * self.noise_std
        return tensor + noise
    
    def generate_batch(
        self,
        batch_size: int,
        commercial_ratio: float = 0.1,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate a batch of synthetic plates.
        
        Args:
            batch_size: Number of plates to generate
            commercial_ratio: Ratio of commercial (yellow) plates
            
        Returns:
            Tuple of (images [B, 3, H, W], list of text strings)
        """
        images = []
        texts = []
        
        for i in range(batch_size):
            commercial = random.random() < commercial_ratio
            image, text = self.generate(commercial=commercial)
            images.append(image)
            texts.append(text)
        
        batch = torch.stack(images, dim=0)
        return batch, texts


class SyntheticPlateDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for synthetic license plates.
    
    Generates plates on-the-fly for training.
    """
    
    def __init__(
        self,
        num_samples: int = 50000,
        output_size: Tuple[int, int] = (128, 32),
        commercial_ratio: float = 0.1,
        transform=None,
    ):
        self.num_samples = num_samples
        self.output_size = output_size
        self.commercial_ratio = commercial_ratio
        self.transform = transform
        
        self.generator = SyntheticPlateGenerator(output_size=output_size)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # Generate random plate
        commercial = random.random() < self.commercial_ratio
        image, text = self.generator.generate(commercial=commercial)
        
        if self.transform:
            image = self.transform(image)
        
        return image, text


def create_ocr_training_data(
    num_samples: int = 50000,
    output_dir: str = "synthetic_plates",
    batch_size: int = 100,
):
    """
    Generate and save synthetic OCR training data.
    
    Args:
        num_samples: Total number of samples to generate
        output_dir: Directory to save images and labels
        batch_size: Number of samples per file
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticPlateGenerator()
    
    all_labels = []
    
    for batch_idx in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - batch_idx)
        images, texts = generator.generate_batch(current_batch_size)
        
        # Save batch
        batch_file = os.path.join(output_dir, f"batch_{batch_idx:06d}.pt")
        torch.save({'images': images, 'texts': texts}, batch_file)
        
        all_labels.extend(texts)
        
        if (batch_idx + batch_size) % 1000 == 0:
            print(f"Generated {batch_idx + batch_size}/{num_samples} plates")
    
    # Save all labels
    labels_file = os.path.join(output_dir, "labels.txt")
    with open(labels_file, 'w') as f:
        for text in all_labels:
            f.write(f"{text}\n")
    
    print(f"✓ Generated {num_samples} synthetic plates in {output_dir}")


if __name__ == "__main__":
    print("📊 Synthetic Plate Generator Test\n")
    
    # Test single generation
    generator = SyntheticPlateGenerator()
    
    print("Generating sample plates...")
    for i in range(5):
        image, text = generator.generate()
        print(f"  {i+1}. {text} - shape: {image.shape}")
    
    # Test batch generation
    print("\nGenerating batch...")
    batch, texts = generator.generate_batch(10)
    print(f"  Batch shape: {batch.shape}")
    print(f"  Sample texts: {texts[:3]}")
    
    # Test dataset
    print("\nTesting dataset...")
    dataset = SyntheticPlateDataset(num_samples=100)
    print(f"  Dataset length: {len(dataset)}")
    image, text = dataset[0]
    print(f"  Sample: {text}, shape: {image.shape}")
    
    # Test random plate text generation
    print("\nSample plate texts:")
    for _ in range(10):
        print(f"  {generate_plate_text()}")
    
    print("\n✓ Synthetic plate generator test passed!")
