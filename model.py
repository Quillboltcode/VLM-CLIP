# main.py
import torch
from transformers.models.clip import CLIPModel
import torch.nn as nn
from adapter.peclip import TextualAdapter, TemporalDynamicAdapter, SharedAdapter


class CLIPWithAdapters(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", adapter_dim=64):
        super(CLIPWithAdapters, self).__init__()

        # Load pretrained CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        # Freeze CLIP base parameters (optional)
        for param in self.clip.parameters():
            param.requires_grad = False

        # Get embedding dimension
        self.text_dim = self.clip.text_embed_dim
        self.vision_dim = self.clip.vision_embed_dim

        # Add adapters
        self.text_adapter = TextualAdapter(self.text_dim, adapter_dim)
        self.video_adapter = TemporalDynamicAdapter(self.vision_dim, adapter_dim)
        self.shared_adapter = SharedAdapter(self.vision_dim, num_heads=8)

        # Similarity head
        self.similarity_head = nn.Linear(self.vision_dim, 1)

    def forward(self, input_ids, pixel_values):
        # Get original CLIP embeddings
        text_outputs = self.clip.text_model(input_ids=input_ids)
        video_outputs = self.clip.vision_model(pixel_values=pixel_values)

        text_emb = text_outputs[1]  # pooled output [batch_size, text_dim]
        video_emb = video_outputs[1]  # pooled output [batch_size, vision_dim]

        # Expand embeddings for sequence processing (if needed)
        text_emb = text_emb.unsqueeze(1)  # [batch, 1, dim]
        video_emb = video_emb.unsqueeze(1)

        # Apply adapters
        adapted_text = self.text_adapter(text_emb)
        adapted_video = self.video_adapter(video_emb)
        shared_text = self.shared_adapter(adapted_text)
        shared_video = self.shared_adapter(adapted_video)

        # Compute similarity
        combined = shared_text * shared_video
        logits = self.similarity_head(combined).squeeze(-1)

        return logits

    # Add these methods to the CLIPWithAdapters class

    def save_adapter_weights(self, path):
        """Save only the adapter parameters."""
        torch.save(
            {
                "text_adapter": self.text_adapter.state_dict(),
                "video_adapter": self.video_adapter.state_dict(),
                "shared_adapter": self.shared_adapter.state_dict(),
                "classifier": self.similarity_head.state_dict(),
            },
            path,
        )

    def load_adapter_weights(self, path):
        """Load only the adapter parameters."""
        checkpoint = torch.load(path)
        self.text_adapter.load_state_dict(checkpoint["text_adapter"])
        self.video_adapter.load_state_dict(checkpoint["video_adapter"])
        self.shared_adapter.load_state_dict(checkpoint["shared_adapter"])
        self.similarity_head.load_state_dict(checkpoint["classifier"])
