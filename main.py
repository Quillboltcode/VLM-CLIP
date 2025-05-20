# main.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder  # For image datasets
from transformers.models.clip import CLIPTokenizerFast
from model import CLIPWithAdapters


# Define dataset (replace with your own)
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, images, labels):
        self.texts = texts
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": self.texts[idx],
            "pixel_values": self.images[idx],
            "labels": self.labels[idx],
        }

    def __len__(self):
        return len(self.labels)


# Training loop
def train(model, train_loader, val_loader, epochs=5, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        # Validation (optional)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, pixel_values)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# Example usage
if __name__ == "__main__":
    # Initialize model and data
    model = CLIPWithAdapters(num_classes=7)
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Dummy data (replace with real data)
    texts = ["a happy person", "an angry face"] * 100
    images = torch.rand(200, 3, 224, 224)
    labels = torch.randint(0, 7, (200,))

    dataset = EmotionDataset(texts, images, labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    # Train
    train(model, train_loader, val_loader, epochs=3)
