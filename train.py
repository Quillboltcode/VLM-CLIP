# train.py

import torch
from torch.optim import AdamW
from transformers.models.clip import CLIPProcessor
from model import CLIPWithAdapters
from config import AdapterConfig


def create_optimizer(model, lr=1e-4):
    adapter_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(adapter_params, lr=lr)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_config = AdapterConfig(
        context_layers=2, textual_layers=2, shared_layer_index=6
    )

    # Load processor and model
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPWithAdapters.from_pretrained(
        model_name, use_adapters=True, adapter_config=adapter_config
    ).to(device)

    optimizer = create_optimizer(model)

    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} | Shape: {param.shape}")

    # Dummy training loop
    model.train()
    dummy_image = torch.rand(2, 3, 224, 224).to(device)
    outputs = model.get_image_features(dummy_image)
    loss = outputs.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
    train()
