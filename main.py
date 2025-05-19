# main.py

from transformers.models.clip import CLIPTokenizerFast, CLIPProcessor
import torch
from model import CLIPWithAdapters


def main():
    # Initialize model and tokenizer
    model = CLIPWithAdapters()
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Dummy inputs
    texts = ["a happy person", "an angry face"]
    images = torch.rand(2, 3, 224, 224)  # dummy images

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Forward pass
    logits = model(inputs["input_ids"], images)
    print("Logits:", logits)


if __name__ == "__main__":
    main()
