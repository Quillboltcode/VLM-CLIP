# dataset.py
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import config # Import your config file

class BaseEnhancedFolderDataset(Dataset):
    def __init__(self, root_dir, clip_processor, vlm_context_extractor, mode='test', max_images=None, emotions=config.EMOTIONS, device=config.DEVICE):
        self.root_dir = root_dir
        self.clip_processor = clip_processor
        self.vlm_context_extractor = vlm_context_extractor
        self.mode = mode
        self.emotions = emotions
        self.device = device # For creating fallback tensors

        self.images = []
        self.labels = []
        self.image_paths = []

        self._load_data(max_images)
        print(f"Loaded {len(self.images)} images for {mode} mode from {root_dir}")

    def _load_data(self, max_images):
        for emotion_idx, emotion_name in enumerate(self.emotions):
            emotion_folder = os.path.join(self.root_dir, emotion_name)
            if not os.path.exists(emotion_folder):
                print(f"Warning: {emotion_folder} does not exist!")
                continue

            img_files = [f for f in os.listdir(emotion_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(img_files) # Shuffle before limiting

            if max_images:
                # Distribute max_images somewhat evenly if specified
                max_per_class = max(1, max_images // len(self.emotions)) # Ensure at least 1 if max_images is small
                img_files = img_files[:max_per_class]

            for img_file in img_files:
                img_path = os.path.join(emotion_folder, img_file)
                self.images.append(img_path) # Store path, load image in __getitem__
                self.labels.append(emotion_idx)
                self.image_paths.append(img_path) # Keep original path for reference

    def __len__(self):
        return len(self.images)

    def _get_common_item(self, idx):
        img_path = self.images[idx]
        emotion_idx = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}. Returning dummy data.")
            # Return dummy data that matches expected types and shapes
            dummy_pixel_values = torch.zeros((3, 224, 224)) # Adjust size if your processor uses different
            dummy_label = torch.tensor(0, dtype=torch.long) # Or a default label
            dummy_context_features = torch.zeros(self.vlm_context_extractor.clip_model.text_model.config.hidden_size if hasattr(self.vlm_context_extractor, 'clip_model') else 768, device=self.device) # Fallback dim for CLIP L/14
            return dummy_pixel_values, dummy_label, img_path, dummy_context_features, "Error loading image"


        # Process image for CLIP
        # The processor typically handles normalization and resizing.
        processed_image = self.clip_processor(images=image, return_tensors="pt", padding=True)
        pixel_values = processed_image['pixel_values'][0] # Remove batch dimension

        # Extract and encode context using VLM
        context_description = self.vlm_context_extractor.extract_context_from_image(image)
        context_features = self.vlm_context_extractor.encode_context_descriptions(context_description)

        if context_features is None:
            # Fallback to a zero tensor if context extraction or encoding fails
            # Use the text feature dimension of the CLIP model used by the VLM context extractor
            # or a common default like 768 for CLIP ViT-L/14
            feature_dim = self.vlm_context_extractor.clip_model.text_model.config.hidden_size if hasattr(self.vlm_context_extractor, 'clip_model') else 768
            context_features = torch.zeros(feature_dim, device=self.device)
        else:
            context_features = context_features[0] # Remove batch dimension

        return pixel_values, torch.tensor(emotion_idx, dtype=torch.long), img_path, context_features, context_description


class EnhancedFolderDataset(BaseEnhancedFolderDataset):
    """Dataset for training, returns (pixel_values, label, img_path, context_features)."""
    def __getitem__(self, idx):
        pixel_values, label, img_path, context_features, _ = self._get_common_item(idx)
        return pixel_values, label, img_path, context_features


class EnhancedFolderDatasetWithContext(BaseEnhancedFolderDataset):
    """Dataset for evaluation, returns (pixel_values, label, img_path, context_features, context_description_text)."""
    def __getitem__(self, idx):
        pixel_values, label, img_path, context_features, context_description = self._get_common_item(idx)
        return (
            pixel_values,
            label,
            img_path,
            context_features,
            context_description if context_description else "No description available"
        )