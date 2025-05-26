# model.py

import torch
from torch import nn, optim
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from constants import get_emotion_descriptions, EMOTIONS # Assuming constants.py is in the same directory

# Set global device (can be moved to main.py or a config file if preferred)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualAdapter(nn.Module):
    """Adapter module for the visual branch of CLIP"""
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TextAdapter(nn.Module):
    """Adapter module for the text branch of CLIP"""
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class CLIPAdapter:
    """CLIP-Adapter: Fine-tuning CLIP with bottleneck adapters for few-shot learning"""

    def __init__(self, model_name, alpha=0.2, beta=0.2, bottleneck_dim=64):
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze the CLIP backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Get feature dimensions
        with torch.no_grad():
            # Get a dummy image to determine feature dimensions
            dummy_image = torch.zeros(1, 3, 224, 224).to(device)
            image_features = self.model.get_image_features(pixel_values=dummy_image)
            self.image_feature_dim = image_features.shape[-1]

            # Get text feature dimension (same as image feature dimension in CLIP)
            self.text_feature_dim = self.image_feature_dim

        # Initialize adapters
        self.visual_adapter = VisualAdapter(self.image_feature_dim, bottleneck_dim).to(device)
        self.text_adapter = TextAdapter(self.text_feature_dim, bottleneck_dim).to(device)

        # Set residual ratios
        self.alpha = alpha  # Visual branch residual ratio
        self.beta = beta    # Text branch residual ratio

        # Generate emotion descriptions
        self.emotion_descriptions = get_emotion_descriptions()

        # Pre-encode text descriptions
        self.encode_emotion_descriptions()

    def encode_emotion_descriptions(self):
        """Pre-compute text embeddings for each emotion description"""
        print("Encoding emotion descriptions...")

        self.original_emotion_text_features = {}
        self.emotion_text_features_per_description = {}

        with torch.no_grad():
            for emotion, descriptions in self.emotion_descriptions.items():
                # Store features for each individual description
                self.emotion_text_features_per_description[emotion] = []

                # Process each description individually
                for description in descriptions:
                    # Process text with CLIP processor
                    text_inputs = self.processor(
                        text=[description],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)

                    # Get text features
                    text_outputs = self.model.get_text_features(**text_inputs)

                    # Normalize features
                    text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)

                    # Store individual description feature
                    self.emotion_text_features_per_description[emotion].append(text_features)

                # Stack all descriptions for this emotion
                all_desc_features = torch.cat(self.emotion_text_features_per_description[emotion], dim=0)

                # Store average embedding for this emotion
                self.original_emotion_text_features[emotion] = all_desc_features.mean(dim=0, keepdim=True)

            # Stack all emotion embeddings for batch processing
            self.emotion_embedding_tensor = torch.cat(list(self.original_emotion_text_features.values()), dim=0)

    def update_emotion_embeddings(self):
        """Update the emotion embeddings using the text adapter"""
        with torch.no_grad():
            # Apply text adapter to original emotion text features
            adapted_features = {}
            for emotion, features in self.original_emotion_text_features.items():
                # Apply the adapter
                adapter_output = self.text_adapter(features)

                # Apply residual connection with beta
                adapted_feature = self.beta * adapter_output + (1 - self.beta) * features

                # Normalize the features
                adapted_feature = adapted_feature / adapted_feature.norm(dim=-1, keepdim=True)

                adapted_features[emotion] = adapted_feature

            # Update the emotion embedding tensor for inference
            self.adapted_emotion_embedding_tensor = torch.cat(list(adapted_features.values()), dim=0)

    def train(self, train_loader, num_epochs=50, learning_rate=3e-4):
        """Train the adapter modules"""
        # Set adapters to training mode
        self.visual_adapter.train()
        self.text_adapter.train()

        # Setup optimizer for adapter modules only
        optimizer = optim.Adam(
            list(self.visual_adapter.parameters()) +
            list(self.text_adapter.parameters()),
            lr=learning_rate
        )

        # Temperature parameter from CLIP
        temperature = self.model.logit_scale.exp().item()

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for pixel_values, labels, _ in progress_bar:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

                # Get original CLIP image features
                with torch.no_grad():
                    original_image_features = self.model.get_image_features(pixel_values=pixel_values)
                    original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)

                # Apply visual adapter
                adapted_image_features = self.visual_adapter(original_image_features)

                # Apply residual connection with alpha
                final_image_features = self.alpha * adapted_image_features + (1 - self.alpha) * original_image_features

                # Normalize features
                final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)

                # Get original text features for each class
                original_text_features = self.emotion_embedding_tensor.clone().detach()

                # Apply text adapter
                adapted_text_features = self.text_adapter(original_text_features)

                # Apply residual connection with beta
                final_text_features = self.beta * adapted_text_features + (1 - self.beta) * original_text_features

                # Normalize features
                final_text_features = final_text_features / final_text_features.norm(dim=-1, keepdim=True)

                # Compute logits
                logits = temperature * torch.matmul(final_image_features, final_text_features.T)

                # Compute contrastive loss (cross-entropy)
                loss = nn.CrossEntropyLoss()(logits, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # Update progress bar
                progress_bar.set_postfix({"Loss": f"{total_loss/batch_count:.4f}"})

            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Update emotion embeddings for evaluation
            self.update_emotion_embeddings()

        # Final update of emotion embeddings
        self.update_emotion_embeddings()

        # Set adapters to evaluation mode
        self.visual_adapter.eval()
        self.text_adapter.eval()

    def predict(self, pixel_values):
        """Predict emotion from image pixel values using the adapted model"""
        with torch.no_grad():
            # Get original image features
            original_image_features = self.model.get_image_features(pixel_values=pixel_values)
            original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)

            # Apply visual adapter
            if hasattr(self, 'visual_adapter'):
                adapted_image_features = self.visual_adapter(original_image_features)

                # Apply residual connection with alpha
                final_image_features = self.alpha * adapted_image_features + (1 - self.alpha) * original_image_features

                # Normalize features
                final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)
            else:
                final_image_features = original_image_features

            # Use adapted emotion embeddings if available
            if hasattr(self, 'adapted_emotion_embedding_tensor'):
                similarity = 100 * torch.matmul(final_image_features, self.adapted_emotion_embedding_tensor.transpose(0, 1))
            else: # Fallback for initial or un-adapted state
                similarity = 100 * torch.matmul(final_image_features, self.emotion_embedding_tensor.transpose(0, 1))


            # Get predicted class
            probs = torch.softmax(similarity, dim=1)

        return probs

    def predict_with_all_descriptions(self, pixel_values):
        """Predict emotion using all individual descriptions and aggregate results"""
        with torch.no_grad():
            # Get original image features
            original_image_features = self.model.get_image_features(pixel_values=pixel_values)
            original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)

            # Apply visual adapter
            if hasattr(self, 'visual_adapter'):
                adapted_image_features = self.visual_adapter(original_image_features)
                final_image_features = self.alpha * adapted_image_features + (1 - self.alpha) * original_image_features
                final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)
            else:
                final_image_features = original_image_features

            # Calculate similarities for each emotion using all descriptions
            emotion_scores = []

            for emotion_idx, emotion in enumerate(EMOTIONS):
                # Get all descriptions for this emotion
                description_features = self.emotion_text_features_per_description[emotion]

                # Apply text adapter to each description
                adapted_desc_features = []
                for desc_feature in description_features:
                    # Apply text adapter
                    adapted_feature = self.text_adapter(desc_feature.to(device)) # Ensure it's on device

                    # Apply residual connection
                    final_feature = self.beta * adapted_feature + (1 - self.beta) * desc_feature.to(device)

                    # Normalize
                    final_feature = final_feature / final_feature.norm(dim=-1, keepdim=True)
                    adapted_desc_features.append(final_feature)

                # Calculate similarity with each description
                desc_similarities = []
                for desc_feature in adapted_desc_features:
                    similarity = 100 * torch.matmul(final_image_features, desc_feature.transpose(0, 1))
                    desc_similarities.append(similarity)

                # Stack all description similarities
                all_desc_similarities = torch.cat(desc_similarities, dim=1)

                # Take the maximum similarity across all descriptions for this emotion
                max_similarity, _ = torch.max(all_desc_similarities, dim=1)
                emotion_scores.append(max_similarity)

            # Stack all emotion scores
            all_emotion_scores = torch.stack(emotion_scores, dim=1)

            # Apply softmax to get probabilities
            probs = torch.softmax(all_emotion_scores, dim=1)

        return probs

class ZeroShotEmotionRecognition:
    """Zero-shot emotion recognition using CLIP with detailed descriptions"""

    def __init__(self, model_name):
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Generate emotion descriptions
        self.emotion_descriptions = get_emotion_descriptions()

        # Pre-encode text descriptions
        self.encode_emotion_descriptions()

    def encode_emotion_descriptions(self):
        """Pre-compute text embeddings for each emotion description"""
        print("Encoding emotion descriptions...")

        self.emotion_text_features = {}
        self.emotion_text_features_per_description = {}

        with torch.no_grad():
            for emotion, descriptions in self.emotion_descriptions.items():
                # Store features for each individual description
                self.emotion_text_features_per_description[emotion] = []

                # Process each description individually
                for description in descriptions:
                    # Process text with CLIP processor
                    text_inputs = self.processor(
                        text=[description],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)

                    # Get text features
                    text_outputs = self.model.get_text_features(**text_inputs)

                    # Normalize features
                    text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)

                    # Store individual description feature
                    self.emotion_text_features_per_description[emotion].append(text_features)

                # Stack all descriptions for this emotion
                all_desc_features = torch.cat(self.emotion_text_features_per_description[emotion], dim=0)

                # Store average embedding for this emotion
                self.emotion_text_features[emotion] = all_desc_features.mean(dim=0, keepdim=True)

            # Stack all emotion embeddings for batch processing
            self.emotion_embedding_tensor = torch.cat(list(self.emotion_text_features.values()), dim=0)

    def predict(self, pixel_values):
        """Predict emotion from image pixel values using average of descriptions"""
        with torch.no_grad():
            # Get image features
            image_features = self.model.get_image_features(pixel_values=pixel_values.to(device))

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity scores with emotion text embeddings (dot product)
            similarity = 100 * torch.matmul(image_features, self.emotion_embedding_tensor.to(device).transpose(0, 1))

            # Get predicted class
            probs = torch.softmax(similarity, dim=1)

        return probs

    def predict_with_all_descriptions(self, pixel_values):
        """Predict emotion using all individual descriptions and aggregate results"""
        with torch.no_grad():
            # Get image features
            image_features = self.model.get_image_features(pixel_values=pixel_values.to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarities for each emotion using all descriptions
            emotion_scores = []

            for emotion_idx, emotion in enumerate(EMOTIONS):
                # Get all descriptions for this emotion
                description_features = self.emotion_text_features_per_description[emotion]

                # Calculate similarity with each description
                desc_similarities = []
                for desc_feature in description_features:
                    similarity = 100 * torch.matmul(image_features, desc_feature.to(device).transpose(0, 1))
                    desc_similarities.append(similarity)

                # Stack all description similarities
                all_desc_similarities = torch.cat(desc_similarities, dim=1)

                # Take the maximum similarity across all descriptions for this emotion
                max_similarity, _ = torch.max(all_desc_similarities, dim=1)
                emotion_scores.append(max_similarity)

            # Stack all emotion scores
            all_emotion_scores = torch.stack(emotion_scores, dim=1)

            # Apply softmax to get probabilities
            probs = torch.softmax(all_emotion_scores, dim=1)

        return probs