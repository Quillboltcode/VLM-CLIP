import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from model import CLIPWithAdapters


class CLIPAdapterTrainer:
    """
    Trainer class for fine-tuning CLIP with adapters.
    """

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=0,
        max_grad_norm=1.0,
        output_dir="./clip_adapter_checkpoints",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get trainable parameters (only adapters)
        self.trainable_params = []
        for name, param in model.named_parameters():
            if "adapter" in name or "shared_adapters" in name:
                self.trainable_params.append(param)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=learning_rate, weight_decay=weight_decay
        )

    def train(self, num_epochs, save_every=1, eval_every=1):
        """
        Train the model for the specified number of epochs.
        """
        device = next(self.model.parameters()).device
        total_steps = len(self.train_dataloader) * num_epochs

        # Create scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0

            with tqdm(
                total=len(self.train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}"
            ) as pbar:
                for batch in self.train_dataloader:
                    # Move batch to device
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    # Forward pass
                    outputs = self.model(
                        input_ids=batch.get("input_ids"),
                        attention_mask=batch.get("attention_mask"),
                        pixel_values=batch.get("pixel_values"),
                        return_loss=True,
                    )

                    loss = outputs["loss"]

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Clip gradients
                    nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)

                    # Update weights
                    self.optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")

            # Validate if requested
            if self.val_dataloader is not None and (epoch + 1) % eval_every == 0:
                val_loss = self.evaluate()
                print(f"Epoch {epoch + 1} - Validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(os.path.join(self.output_dir, "best_adapter"))

            # Save checkpoint if requested
            if (epoch + 1) % save_every == 0:
                self.save_model(
                    os.path.join(self.output_dir, f"adapter_epoch_{epoch + 1}")
                )

        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_adapter"))

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        assert self.val_dataloader is not None, (
            "val_dataloader must not be None to run eval"
        )
        self.model.eval()
        device = next(self.model.parameters()).device
        val_loss = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                outputs = self.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    return_loss=True,
                )

                val_loss += outputs["loss"].item()

        return val_loss / len(self.val_dataloader)

    def save_model(self, path):
        """
        Save adapter weights to the specified path.
        """
        self.model.save_adapter_weights(f"{path}.pt")

    def load_model(self, path):
        """
        Load adapter weights from the specified path.
        """
        self.model.load_adapter_weights(f"{path}.pt")
