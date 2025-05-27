# main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import os
import logging
from datetime import datetime
import numpy as np
import random

# Import from your custom modules
import config
from model_v import EnhancedCLIPAdapter, VLMContextExtractor
from dataset.enhance import EnhancedFolderDataset, EnhancedFolderDatasetWithContext
from utils import (
    evaluate_enhanced_model,
    display_results_with_contexts,
    analyze_context_quality,
)

seed = config.SEED  # Or any other integer
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# For CUDNN
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# Create a timestamp for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),  # This will also print to console
    ],
)

logger = logging.getLogger(__name__)
logger.info("Starting Enhanced CLIP-Adapter workflow...")


def train_model(model, train_loader, num_epochs, learning_rate, device):
    """Trains the adapter layers of the EnhancedCLIPAdapter model."""
    model.train()  # Set model to training mode (affects dropout, etc.)

    # Ensure only adapter parameters are optimized
    optimizer = optim.Adam(model.get_trainable_parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()  # Standard loss for classification

    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for pixel_values, labels, _, context_features in progress_bar:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            context_features = context_features.to(device)

            optimizer.zero_grad()

            # Forward pass through the model to get logits
            # The model's forward method should handle the adapters internally
            logits = model(
                pixel_values, context_features, use_adapters_for_training=True
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"Loss": f"{total_loss / batch_count:.4f}"})

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {avg_loss:.4f}")
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {avg_loss:.4f}"
        )

        # Update the adapted emotion embeddings after each epoch of training text_adapter
        model.update_emotion_embeddings()

    model.eval()  # Set model back to evaluation mode after training
    print("Training complete.")


def main_enhanced_workflow():
    print(f"Using device: {config.DEVICE}")

    # Initialize VLM Context Extractor (can be shared)
    # This initializes the VLM and its associated CLIP model for encoding descriptions.
    # It's done once and passed around to avoid re-loading heavy models.
    print("Initializing VLM Context Extractor...")
    logging.info("Initializing VLM Context Extractor...")
    vlm_extractor = VLMContextExtractor(
        model_name=config.VLM_MODEL_NAME,
        device=config.DEVICE,  # Pass device to VLM extractor
    )

    # Initialize Enhanced CLIP-Adapter
    print("Initializing Enhanced CLIP-Adapter with VLM context...")
    logging.info("Initializing Enhanced CLIP-Adapter with VLM context...")
    enhanced_adapter_model = EnhancedCLIPAdapter(
        clip_model_name=config.CLIP_MODEL_NAME,
        alpha=config.ALPHA,
        beta=config.BETA,
        gamma=config.GAMMA,
        bottleneck_dim=config.BOTTLENECK_DIM,
        device=config.DEVICE,
        vlm_context_extractor=vlm_extractor,  # Pass the pre-initialized extractor
    )
    enhanced_adapter_model.encode_emotion_descriptions(emotions=config.EMOTIONS)
    enhanced_adapter_model.print_model_structure()

    # Create datasets
    print("Creating enhanced datasets with VLM context extraction...")
    # For training, we need context features but not necessarily the raw text.
    train_dataset = EnhancedFolderDataset(
        root_dir=config.TRAIN_DIR,
        clip_processor=enhanced_adapter_model.processor,  # Use CLIP processor from main model
        vlm_context_extractor=vlm_extractor,
        mode="train",
        max_images=200,  # Limit for faster example run, adjust as needed
        emotions=config.EMOTIONS,
        device=config.DEVICE,
    )
    # For testing, we also want the raw context text for analysis.
    test_dataset = EnhancedFolderDatasetWithContext(
        root_dir=config.TEST_DIR,
        clip_processor=enhanced_adapter_model.processor,
        vlm_context_extractor=vlm_extractor,
        mode="test",
        max_images=50,  # Limit for faster example run
        emotions=config.EMOTIONS,
        device=config.DEVICE,
    )

    if (
        not train_dataset
        or not test_dataset
        or len(train_dataset) == 0
        or len(test_dataset) == 0
    ):
        print(
            "Failed to load datasets or datasets are empty. Please check paths and data."
        )
        return

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )  # num_workers > 0 can speed up
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"Train set: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Test set: {len(test_dataset)} samples, {len(test_loader)} batches")

    # Train the model
    print("Training Enhanced CLIP-Adapter...")
    train_model(
        enhanced_adapter_model,
        train_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        device=config.DEVICE,
    )

    # Save the trained adapter weights (optional)
    torch.save(
        {
            "visual_adapter_state_dict": enhanced_adapter_model.visual_adapter.state_dict(),
            "text_adapter_state_dict": enhanced_adapter_model.text_adapter.state_dict(),
            "context_adapter_state_dict": enhanced_adapter_model.context_adapter.state_dict(),
        },
        "enhanced_adapters_weights.pth",
    )
    logging.info("Trained adapter weights saved to enhanced_adapters_weights.pth")
    # print("Trained adapter weights saved to enhanced_adapters_weights.pth")

    # Evaluate the model
    print("Evaluating Enhanced CLIP-Adapter...")
    # The model should be in eval mode after training, but ensure it.
    enhanced_adapter_model.eval()
    # Ensure adapted emotion embeddings are up-to-date for evaluation
    enhanced_adapter_model.update_emotion_embeddings()

    evaluation_results = evaluate_enhanced_model(
        enhanced_adapter_model,
        test_loader,
        device=config.DEVICE,
        emotions=config.EMOTIONS,
    )

    # Display comprehensive results
    display_results_with_contexts(
        evaluation_results,
        emotions=config.EMOTIONS,
        num_samples_per_class=config.NUM_SAMPLES_TO_DISPLAY_PER_CLASS,
    )

    # Analyze context quality from the evaluation results
    # evaluation_results tuple: (accuracy, conf_matrix, class_report, all_preds, all_labels,
    #                            all_image_paths, all_confidences, all_similarity_scores, all_contexts_text)
    # all_contexts_text is at index 8, all_labels is at index 4
    analyze_context_quality(
        all_contexts_text=evaluation_results[8],
        all_labels=evaluation_results[4],
        emotions=config.EMOTIONS,
    )


if __name__ == "__main__":
    main_enhanced_workflow()

