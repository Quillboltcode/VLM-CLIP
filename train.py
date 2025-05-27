# train.py

import os
import torch
from transformers import CLIPProcessor
from model_m import CLIPWithAdapters
from dataset import RAFDBDataset, create_dataloaders
from trainer import CLIPAdapterTrainer
from pathlib import Path

def main():
    # Configuration
    config = {
        # Model settings
        "clip_model_name": "openai/clip-vit-base-patch32",
        "text_adapter_size": 256,
        "vision_adapter_size": 256,
        "shared_adapter_layers": 2,
        
        # Training settings
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        
        # Data settings
        "data_root": "/path/to/RAFDB",  # Update this path
        "max_length": 77,
        
        # Output settings
        "output_dir": "./clip_adapter_checkpoints",
        "save_every": 1,
        "eval_every": 1,
    }

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize CLIP processor
    processor = CLIPProcessor.from_pretrained(config["clip_model_name"])

    # Create dataloaders
    dataloaders = create_dataloaders(
        root_dir=config["data_root"],
        processor=processor,
        batch_size=config["batch_size"]
    )

    # Initialize model
    model = CLIPWithAdapters(
        clip_model_name=config["clip_model_name"],
        text_adapter_size=config["text_adapter_size"],
        vision_adapter_size=config["vision_adapter_size"],
        shared_adapter_layers=config["shared_adapter_layers"],
        freeze_clip=True,  # Freeze base CLIP model
        use_text_adapter=True,
        use_vision_adapter=True,
        use_shared_adapters=True
    )
    model = model.to(device)

    # Initialize trainer
    trainer = CLIPAdapterTrainer(
        model=model,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['val'],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        max_grad_norm=config["max_grad_norm"],
        output_dir=config["output_dir"]
    )

    # Train model
    print("Starting training...")
    trainer.train(
        num_epochs=config["num_epochs"],
        save_every=config["save_every"],
        eval_every=config["eval_every"]
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    test_loss = trainer.evaluate()
    print(f"Final test loss: {test_loss:.4f}")

    # Save final model
    final_save_path = os.path.join(config["output_dir"], "final_model")
    trainer.save_model(final_save_path)
    print(f"Model saved to {final_save_path}")

if __name__ == "__main__":
    main()
