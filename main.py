# main.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader

# Import from other modules
import constants
from data_utils import FolderDataset
from model import CLIPAdapter, ZeroShotEmotionRecognition, device # Import device from model.py
from evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_class_similarities,
    visualize_predictions
)

def test_single_image(model, image_path, use_all_descriptions=False):
    """Test the model on a single image"""

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file {image_path} not found.")
        return None, None, None

    inputs = model.processor(
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Get prediction
    model.model.eval() # Ensure CLIP model is in eval mode
    if hasattr(model, 'visual_adapter') and model.visual_adapter is not None:
        model.visual_adapter.eval()
    if hasattr(model, 'text_adapter') and model.text_adapter is not None:
        model.text_adapter.eval()


    with torch.no_grad():
        if use_all_descriptions and hasattr(model, 'predict_with_all_descriptions'):
            probs = model.predict_with_all_descriptions(inputs.pixel_values)[0]
        else:
            probs = model.predict(inputs.pixel_values)[0]

        confidence, predicted = torch.max(probs, 0)

    # Get emotion label
    emotion = constants.EMOTIONS[predicted.item()]

    # Display image with prediction
    plt.figure(figsize=(6, 8))
    plt.imshow(image)
    plt.title(f"Predicted: {emotion} ({confidence.item():.2f})")
    plt.axis('off')
    plt.show()

    # Show emotion probabilities
    plt.figure(figsize=(10, 5))
    sns.barplot(x=constants.EMOTIONS, y=probs.cpu().numpy())
    plt.title('Emotion Similarity Scores')
    plt.ylabel('Similarity Score')
    plt.xlabel('Emotion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print emotion probabilities
    print("\nEmotion similarity scores:")
    for i, emotion_name in enumerate(constants.EMOTIONS):
        print(f"{emotion_name}: {probs[i].item():.4f}")

    # Print text descriptions used for the predicted emotion
    print(f"\nText descriptions used for '{emotion}':")
    if emotion in model.emotion_descriptions:
        for i, desc in enumerate(model.emotion_descriptions[emotion]):
            print(f"{i+1}. {desc}")
    else:
        print(f"No descriptions found for emotion: {emotion}")


    return emotion, confidence.item(), probs.cpu().numpy()

def compare_models(clip_adapter, zero_shot_model, test_loader, use_all_descriptions=True):
    """Compare the performance of CLIP-Adapter and Zero-Shot models with different description strategies"""

    # Evaluate Zero-Shot model with averaged descriptions
    print("\nEvaluating Zero-Shot model with averaged descriptions...")
    zero_shot_results = evaluate_model(zero_shot_model, test_loader, use_all_descriptions=False)
    zero_shot_accuracy = zero_shot_results[0]

    # Evaluate Zero-Shot model with all descriptions
    zero_shot_all_desc_accuracy = 0.0 # Initialize
    if use_all_descriptions:
        print("\nEvaluating Zero-Shot model with all descriptions...")
        zero_shot_all_desc_results = evaluate_model(zero_shot_model, test_loader, use_all_descriptions=True)
        zero_shot_all_desc_accuracy = zero_shot_all_desc_results[0]

    # Evaluate CLIP-Adapter with averaged descriptions
    print("\nEvaluating CLIP-Adapter with averaged descriptions...")
    adapter_results = evaluate_model(clip_adapter, test_loader, use_all_descriptions=False)
    adapter_accuracy = adapter_results[0]

    # Evaluate CLIP-Adapter with all descriptions
    adapter_all_desc_accuracy = 0.0 # Initialize
    if use_all_descriptions:
        print("\nEvaluating CLIP-Adapter with all descriptions...")
        adapter_all_desc_results = evaluate_model(clip_adapter, test_loader, use_all_descriptions=True)
        adapter_all_desc_accuracy = adapter_all_desc_results[0]

    # Plot confusion matrices
    if zero_shot_results[1] is not None and zero_shot_results[1].size > 0 :
        plot_confusion_matrix(zero_shot_results[1], "Zero-Shot Confusion Matrix (Avg Descriptions)")
    if use_all_descriptions and zero_shot_all_desc_results[1] is not None and zero_shot_all_desc_results[1].size > 0:
        plot_confusion_matrix(zero_shot_all_desc_results[1], "Zero-Shot Confusion Matrix (All Descriptions)")
    if adapter_results[1] is not None and adapter_results[1].size > 0:
        plot_confusion_matrix(adapter_results[1], "CLIP-Adapter Confusion Matrix (Avg Descriptions)")
    if use_all_descriptions and adapter_all_desc_results[1] is not None and adapter_all_desc_results[1].size > 0:
        plot_confusion_matrix(adapter_all_desc_results[1], "CLIP-Adapter Confusion Matrix (All Descriptions)")

    # Plot class similarities
    plot_class_similarities(zero_shot_results[7], zero_shot_results[4], "Zero-Shot Similarity (Avg Descriptions)")
    if use_all_descriptions:
        plot_class_similarities(zero_shot_all_desc_results[7], zero_shot_all_desc_results[4], "Zero-Shot Similarity (All Descriptions)")
    plot_class_similarities(adapter_results[7], adapter_results[4], "CLIP-Adapter Similarity (Avg Descriptions)")
    if use_all_descriptions:
        plot_class_similarities(adapter_all_desc_results[7], adapter_all_desc_results[4], "CLIP-Adapter Similarity (All Descriptions)")

    # Visualize predictions
    visualize_predictions(zero_shot_results[5], zero_shot_results[4], zero_shot_results[3], zero_shot_results[6],
                        num_examples=5, title="Zero-Shot Predictions (Avg Descriptions)")
    if use_all_descriptions:
        visualize_predictions(zero_shot_all_desc_results[5], zero_shot_all_desc_results[4], zero_shot_all_desc_results[3],
                            zero_shot_all_desc_results[6], num_examples=5, title="Zero-Shot Predictions (All Descriptions)")
    visualize_predictions(adapter_results[5], adapter_results[4], adapter_results[3], adapter_results[6],
                        num_examples=5, title="CLIP-Adapter Predictions (Avg Descriptions)")
    if use_all_descriptions:
        visualize_predictions(adapter_all_desc_results[5], adapter_all_desc_results[4], adapter_all_desc_results[3],
                            adapter_all_desc_results[6], num_examples=5, title="CLIP-Adapter Predictions (All Descriptions)")

    # Print results
    print("\n=== Performance Comparison ===")
    print(f"Zero-Shot Accuracy (Avg Descriptions): {zero_shot_accuracy:.4f}")
    if use_all_descriptions:
        print(f"Zero-Shot Accuracy (All Descriptions): {zero_shot_all_desc_accuracy:.4f}")
    print(f"CLIP-Adapter Accuracy (Avg Descriptions): {adapter_accuracy:.4f}")
    if use_all_descriptions:
        print(f"CLIP-Adapter Accuracy (All Descriptions): {adapter_all_desc_accuracy:.4f}")

    # Calculate improvements
    print("\n=== Improvements ===")
    print(f"Adapter vs Zero-Shot (Avg Descriptions): {(adapter_accuracy - zero_shot_accuracy) * 100:.2f}%")
    if use_all_descriptions:
        print(f"Adapter vs Zero-Shot (All Descriptions): {(adapter_all_desc_accuracy - zero_shot_all_desc_accuracy) * 100:.2f}%")
        print(f"All Descriptions vs Avg (Zero-Shot): {(zero_shot_all_desc_accuracy - zero_shot_accuracy) * 100:.2f}%")
        print(f"All Descriptions vs Avg (CLIP-Adapter): {(adapter_all_desc_accuracy - adapter_accuracy) * 100:.2f}%")
        print(f"Best performance - CLIP-Adapter (All Descriptions): {adapter_all_desc_accuracy:.4f}")

    print("\n=== Zero-Shot Classification Report (Avg Descriptions) ===")
    print(zero_shot_results[2])

    if use_all_descriptions:
        print("\n=== Zero-Shot Classification Report (All Descriptions) ===")
        print(zero_shot_all_desc_results[2])

    print("\n=== CLIP-Adapter Classification Report (Avg Descriptions) ===")
    print(adapter_results[2])

    if use_all_descriptions:
        print("\n=== CLIP-Adapter Classification Report (All Descriptions) ===")
        print(adapter_all_desc_results[2])

    # Return the results
    results = {
        "zero_shot_accuracy_avg": zero_shot_accuracy,
        "adapter_accuracy_avg": adapter_accuracy
    }

    if use_all_descriptions:
        results.update({
            "zero_shot_accuracy_all": zero_shot_all_desc_accuracy,
            "adapter_accuracy_all": adapter_all_desc_accuracy
        })

    return results


def main():
    print(f"Using device: {device}") # device is imported from model.py

    # Initialize zero-shot emotion recognition model for comparison
    print(f"Initializing zero-shot emotion recognition with {constants.MODEL_NAME}...")
    zero_shot_model = ZeroShotEmotionRecognition(constants.MODEL_NAME)

    # Initialize CLIP-Adapter model
    print(f"Initializing CLIP-Adapter with {constants.MODEL_NAME}...")
    clip_adapter = CLIPAdapter(
        constants.MODEL_NAME,
        alpha=constants.ALPHA,
        beta=constants.BETA,
        bottleneck_dim=constants.BOTTLENECK_DIM
    )

    # Create datasets
    train_dataset = FolderDataset(constants.TRAIN_DIR, clip_adapter.processor, mode='train')
    test_dataset = FolderDataset(constants.TEST_DIR, clip_adapter.processor, mode='test')

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=2)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # Train CLIP-Adapter
    if len(train_dataset) > 0 :
        print("Training CLIP-Adapter...")
        clip_adapter.train(train_loader, num_epochs=constants.NUM_EPOCHS, learning_rate=constants.LEARNING_RATE)
    else:
        print("Skipping training as train dataset is empty.")


    # Compare models with and without all descriptions
    if len(test_dataset) > 0:
        results = compare_models(clip_adapter, zero_shot_model, test_loader, use_all_descriptions=True)
    else:
        print("Skipping model comparison as test dataset is empty.")


    # Option to test on single image
    test_image_prompt = input("Do you want to test on a single image? (y/n): ")
    if test_image_prompt.lower() == 'y':
        image_path = input("Enter path to image: ")
        if os.path.exists(image_path):
            print("\n=== Testing Zero-Shot (Averaged Descriptions) on single image ===")
            test_single_image(zero_shot_model, image_path, use_all_descriptions=False)

            print("\n=== Testing Zero-Shot (All Descriptions) on single image ===")
            test_single_image(zero_shot_model, image_path, use_all_descriptions=True)

            print("\n=== Testing CLIP-Adapter (Averaged Descriptions) on single image ===")
            test_single_image(clip_adapter, image_path, use_all_descriptions=False)

            print("\n=== Testing CLIP-Adapter (All Descriptions) on single image ===")
            test_single_image(clip_adapter, image_path, use_all_descriptions=True)
        else:
            print(f"Error: Image file {image_path} not found.")

if __name__ == "__main__":
    main()