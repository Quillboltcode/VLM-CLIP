# model.py
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.models.clip import CLIPModel, CLIPProcessor
from adapter.clip_adapter import TextAdapter, VisionAdapter, SharedMHSAttentionAdapter


class CLIPWithAdapters(nn.Module):
    """
    CLIP model with text and vision adapters.
    """

    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        text_adapter_size=256,
        vision_adapter_size=256,
        shared_adapter_layers=2,
        freeze_clip=True,
        use_text_adapter=True,
        use_vision_adapter=True,
        use_shared_adapters=True,
    ):
        super().__init__()

        # Load CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Get dimensions
        text_hidden_size = self.clip.text_model.config.hidden_size
        vision_hidden_size = self.clip.vision_model.config.hidden_size

        # Set adapter flags
        self.use_text_adapter = use_text_adapter
        self.use_vision_adapter = use_vision_adapter
        self.use_shared_adapters = use_shared_adapters

        # Add text adapter
        if self.use_text_adapter:
            self.text_adapter = TextAdapter(text_hidden_size, text_adapter_size)

        # Add vision adapter
        if self.use_vision_adapter:
            self.vision_adapter = VisionAdapter(vision_hidden_size, vision_adapter_size)

        
        # Add shared cross-attention adapters
        if self.use_shared_adapters:
            self.shared_adapters = nn.ModuleList(
            [
                SharedMHSAttentionAdapter(text_hidden_size, vision_hidden_size)
                for _ in range(shared_adapter_layers)
            ]
        )

        # Freeze CLIP weights if specified
        if freeze_clip:
            self._freeze_clip_parameters()

    def _freeze_clip_parameters(self):
        """Freeze all parameters in the original CLIP model."""
        for param in self.clip.parameters():
            param.requires_grad = False

    def _unfreeze_clip_parameters(self):
        """Unfreeze all parameters in the original CLIP model."""
        for param in self.clip.parameters():
            param.requires_grad = True

    def get_text_features(self, input_ids, attention_mask):
        """Get text features with adapter."""
        # Get the outputs from CLIP's text encoder
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Process through text adapter
        text_features = text_outputs.last_hidden_state
        if self.use_text_adapter:
            # Apply text adapter
            text_features = self.text_adapter(text_features)



        # Apply shared adapters if enabled
        if self.use_shared_adapters:
            image_features = (
                self.clip.vision_model.embeddings.position_embedding.weight.unsqueeze(0)
            )
            for shared_adapter in self.shared_adapters:
                text_features = shared_adapter(text_features, image_features)

        text_features = text_features[:, 0, :]
        text_features = self.clip.text_projection(text_features)

        return text_features

    def get_image_features(self, pixel_values):
        """Get image features with adapter."""
        # Get the outputs from CLIP's vision encoder
        vision_outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        # Process through vision adapter
        image_features = vision_outputs.last_hidden_state
        if self.use_vision_adapter:
            image_features = self.vision_adapter(image_features)

        # Apply CLIP's projection (this is needed to get the contrastive loss)
        # We take the features corresponding to the [CLS] token
        image_features = image_features[:, 0, :]
        image_features = self.clip.visual_projection(image_features)

        return image_features

    def forward(
        self, input_ids=None, attention_mask=None, pixel_values=None, return_loss=True
    ):
        """
        Forward pass with both text and image inputs.
        """
        # Calculate text features
        if input_ids is not None and attention_mask is not None:
            text_features = self.get_text_features(input_ids, attention_mask)
        else:
            text_features = None

        # Calculate image features
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
        else:
            image_features = None

        # Calculate contrastive loss if required
        if return_loss and text_features is not None and image_features is not None:
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            logit_scale = self.clip.logit_scale.exp()
            logits_per_text = (
                torch.matmul(text_features, image_features.t()) * logit_scale
            )
            logits_per_image = logits_per_text.t()

            # Compute contrastive loss
            batch_size = text_features.shape[0]
            labels = torch.arange(batch_size, device=text_features.device)
            loss_text = F.cross_entropy(logits_per_text, labels)
            loss_image = F.cross_entropy(logits_per_image, labels)
            loss = (loss_text + loss_image) / 2

            return {
                "loss": loss,
                "text_features": text_features,
                "image_features": image_features,
                "logits_per_text": logits_per_text,
                "logits_per_image": logits_per_image,
            }
        else:
            return {
                "text_features": text_features,
                "image_features": image_features,
            }

    def save_adapter_weights(self, save_path):
        """
        Save only the adapter weights, keeping the original CLIP weights untouched.
        """
        adapter_state_dict = {
            "text_adapter": self.text_adapter.state_dict(),
            "vision_adapter": self.vision_adapter.state_dict(),
            "shared_adapters": self.shared_adapters.state_dict(),
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(adapter_state_dict, save_path)
        print(f"Adapter weights saved to {save_path}")

    def load_adapter_weights(self, load_path):
        """
        Load only the adapter weights, keeping the original CLIP weights untouched.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No adapter weights found at {load_path}")

        adapter_state_dict = torch.load(
            load_path, map_location=next(self.parameters()).device
        )

        self.text_adapter.load_state_dict(adapter_state_dict["text_adapter"])
        self.vision_adapter.load_state_dict(adapter_state_dict["vision_adapter"])
        self.shared_adapters.load_state_dict(adapter_state_dict["shared_adapters"])

        print(f"Adapter weights loaded from {load_path}")
