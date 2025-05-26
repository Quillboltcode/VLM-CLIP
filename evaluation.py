# evaluation.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from constants import EMOTIONS # Assuming constants.py is in the same directory
# Assuming device is defined in model.py or globally accessible, otherwise pass it as an argument or define here
# from model import device # Or define: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, test_loader, use_all_descriptions=False):
    """Evaluate the model on the test set"""

    all_preds = []
    all_labels = []
    all_image_paths = []
    all_confidences = []
    all_similarity_scores = []

    model.model.eval() # Ensure CLIP model is in eval mode
    if hasattr(model, 'visual_adapter'):
        model.visual_adapter.eval()
    if hasattr(model, 'text_adapter'):
        model.text_adapter.eval()


    with torch.no_grad():
        for pixel_values, labels, img_paths in tqdm(test_loader, desc="Testing"):
            pixel_values = pixel_values.to(device)

            # Get predictions using all descriptions if requested
            if use_all_descriptions and hasattr(model, 'predict_with_all_descriptions'):
                probs = model.predict_with_all_descriptions(pixel_values)
            else:
                probs = model.predict(pixel_values)

            confidences, predicted = torch.max(probs, 1)

            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_image_paths.extend(img_paths)
            all_confidences.extend(confidences.cpu().numpy())
            all_similarity_scores.append(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(
        all_labels, all_preds,
        target_names=EMOTIONS,
        digits=4,
        zero_division=0 # Added to prevent warnings if a class has no predictions
    )

    if all_similarity_scores:
        all_similarity_scores = np.vstack(all_similarity_scores)
    else: # Handle case where test_loader might be empty or no predictions made
        all_similarity_scores = np.array([])


    return accuracy, conf_matrix, class_report, all_preds, all_labels, all_image_paths, all_confidences, all_similarity_scores

def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    """Plot confusion matrix"""

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

def plot_class_similarities(similarity_scores, labels, title="Average Similarity Scores"):
    """Plot average similarity scores for each class"""
    if similarity_scores.size == 0:
        print(f"Skipping plot '{title}' due to empty similarity scores.")
        return

    # Calculate average similarity score for each true class
    class_similarities = []
    for class_idx in range(len(EMOTIONS)):
        # Get indices of samples belonging to this class
        class_indices = [i for i, l in enumerate(labels) if l == class_idx]
        if class_indices:
            # Get average similarity scores for this class
            class_similarity = similarity_scores[class_indices].mean(axis=0)
            class_similarities.append(class_similarity)
        else: # Handle if a class is not present in labels
            class_similarities.append(np.zeros(len(EMOTIONS)))


    # Plot similarities
    plt.figure(figsize=(12, 8))
    x = np.arange(len(EMOTIONS))
    width = 0.1 # Adjust width if too many classes or if it looks crowded

    num_true_classes_plotted = 0
    for i, class_sim in enumerate(class_similarities):
        if np.any(class_sim): # Only plot if there are actual similarities for this true class
             plt.bar(x + num_true_classes_plotted*width, class_sim, width, label=f'True: {EMOTIONS[i]}')
             num_true_classes_plotted +=1


    plt.xlabel('Predicted Emotion Class')
    plt.ylabel('Average Similarity Score')
    plt.title(f'{title} Between True Classes and Text Descriptions')
    if num_true_classes_plotted > 0: # Adjust x-ticks based on how many bars are plotted
        plt.xticks(x + width * (num_true_classes_plotted - 1) / 2, EMOTIONS)
    else:
        plt.xticks(x, EMOTIONS)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()


def visualize_predictions(image_paths, true_labels, predicted_labels, confidences, num_examples=5, title="Prediction Examples"):
    """Visualize some example predictions"""
    if not image_paths:
        print("No images to visualize.")
        return

    # Choose a mix of correct and incorrect predictions
    correct_indices = [i for i, (y, y_pred) in enumerate(zip(true_labels, predicted_labels)) if y == y_pred]
    incorrect_indices = [i for i, (y, y_pred) in enumerate(zip(true_labels, predicted_labels)) if y != y_pred]

    # Select samples
    num_correct = min(num_examples // 2 + num_examples % 2, len(correct_indices))
    num_incorrect = min(num_examples // 2, len(incorrect_indices))

    selected_indices = []
    if num_correct > 0:
        selected_indices.extend(np.random.choice(correct_indices, num_correct, replace=False))


    if num_incorrect > 0:
        selected_indices.extend(np.random.choice(incorrect_indices, num_incorrect, replace=False))

    if not selected_indices:
        print("No examples selected for visualization (not enough correct/incorrect predictions or data).")
        return


    # Plot images with predictions
    # Adjust subplot creation if only one image is selected
    if len(selected_indices) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4)) # Smaller figure for single image
        axes = [axes] # Make it iterable
    else:
        fig, axes = plt.subplots(1, len(selected_indices), figsize=(15, 4))


    for i, idx in enumerate(selected_indices):
        img_path = image_paths[idx]
        true_label = EMOTIONS[true_labels[idx]]
        pred_label = EMOTIONS[predicted_labels[idx]]
        confidence = confidences[idx]

        # Open and display image
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            continue


        # Set title with prediction info
        title_text = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}"
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(title_text, color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()