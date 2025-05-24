import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Union, Tuple
from pathlib import Path


descriptions = {
    "angry": [
        "the image of an angry facial emotion with furrowed brows and clenched teeth",
        "a person expressing anger with narrowed eyes and tightened jaw",
        "a face showing intense frustration and hostility",
        "an irritated facial expression with a glaring stare",
        "a person displaying rage with tensed facial muscles",
    ],
    "disgust": [
        "the image of a disgusted facial emotion with wrinkled nose and raised upper lip",
        "a person expressing revulsion with a grimace and squinted eyes",
        "a face showing strong aversion with curled lip",
        "a nauseated facial expression with furrowed brows",
        "a person displaying distaste with pulled back lips",
    ],
    "fear": [
        "the image of a fearful facial emotion with widened eyes and raised eyebrows",
        "a person expressing terror with a dropped jaw and pulled-back lips",
        "a face showing panic with tense mouth and dilated pupils",
        "a frightened facial expression with raised upper eyelids",
        "a person displaying anxiety with frozen stare and pale complexion",
    ],
    "happy": [
        "the image of a happy facial emotion with upturned mouth corners and crinkled eyes",
        "a person expressing joy with a broad smile and relaxed face",
        "a face showing delight with raised cheeks and visible teeth",
        "a cheerful facial expression with beaming smile and bright eyes",
        "a person displaying pleasure with dimples and lifted cheeks",
    ],
    "neutral": [
        "the image of a neutral facial emotion with relaxed features and natural expression",
        "a person with an emotionless face showing no particular feeling",
        "a face with a balanced expression, neither positive nor negative",
        "a composed facial expression with resting features",
        "a person displaying a calm and unemotional demeanor",
    ],
    "sad": [
        "the image of a sad facial emotion with downturned mouth and drooping eyelids",
        "a person expressing sorrow with furrowed brows and quivering lips",
        "a face showing grief with lowered gaze and compressed lips",
        "a melancholic facial expression with sunken cheeks",
        "a person displaying unhappiness with glazed or teary eyes",
    ],
    "surprise": [
        "the image of a surprised facial emotion with raised eyebrows and widened eyes",
        "a person expressing astonishment with an open mouth and stretched skin",
        "a face showing shock with expanded pupils and heightened alertness",
        "a startled facial expression with dropped jaw and gasping mouth",
        "a person displaying amazement with rounded eyes and lifted brows",
    ],
}


class RAFDBDataset(Dataset):
    """
    Dataset for training CLIP with adapters on RAF-DB.
    
    Folder structure:
    ├── RAFDB
    │   ├── train
    │   ├── val
    │   ├── test
    │   ├── angry
    │   │   ├── angry_1.jpg
    │   ├── happy
    │   │   ├── happy_1.jpg
    ...
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        processor,
        max_length: int = 77,
    ):
        """
        Initialize RAFDBDataset.

        Args:
            root_dir: Root directory containing RAF-DB dataset
            split: One of 'train', 'val', or 'test'
            processor: CLIP processor for tokenizing text and processing images
            max_length: Maximum length for text tokens
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.processor = processor
        self.max_length = max_length
        
        # Get split directory
        self.split_dir = self.root_dir / split
        
        # Validate split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory {self.split_dir} does not exist")
            
        # Get all image paths and their corresponding emotions
        self.samples = []
        for emotion in descriptions.keys():
            emotion_dir = self.split_dir / emotion
            if emotion_dir.exists():
                image_files = list(emotion_dir.glob('*.jpg'))
                self.samples.extend([(str(f), emotion) for f in image_files])

        if not self.samples:
            raise ValueError(f"No images found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item to get

        Returns:
            Dict containing:
                - input_ids: Text token ids
                - attention_mask: Attention mask for text
                - pixel_values: Processed image tensor
                - emotion: Emotion label string
                - caption: Caption text
        """
        image_path, emotion = self.samples[idx]
        
        # Randomly select a description for the emotion
        caption = random.choice(descriptions[emotion])
        
        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")

        # Process inputs
        try:
            encoding = self.processor(
                text=caption,
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        except Exception as e:
            raise ValueError(f"Error processing inputs: {e}")

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add additional information
        encoding['emotion'] = emotion
        encoding['caption'] = caption
        
        return encoding


def create_dataloaders(
    root_dir: str,
    processor,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root_dir: Root directory containing RAF-DB dataset
        processor: CLIP processor
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders

    Returns:
        Dictionary containing dataloaders for each split
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = RAFDBDataset(
            root_dir=root_dir,
            split=split,
            processor=processor
        )
        
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers
        )
    
    return dataloaders
