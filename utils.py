# utils.py

import torch
# utils.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import os
from collections import Counter
import config # Import your config file

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# utils.py

def evaluate_enhanced_model(model, test_loader, device=config.DEVICE, emotions=config.EMOTIONS):
    """Evaluates the enhanced model and collects detailed results including context."""
    model.eval() # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    all_image_paths = []
    all_confidences = []
    all_similarity_scores_list = [] # Use a list to append batch scores
    all_contexts_text = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Model"):
            pixel_values, labels, img_paths, context_features, contexts_text = batch

            pixel_values = pixel_values.to(device)
            # context_features might be a list of tensors if batch collation is complex,
            # or already a stacked tensor. Ensure it's a single tensor on the device.
            if isinstance(context_features, list): # Should not happen with default DataLoader
                context_features = torch.stack(context_features).to(device)
            else:
                context_features = context_features.to(device)

            # Get probability scores from the model
            probs = model.predict_probs(pixel_values, context_features) # Assuming predict_probs is implemented
            confidences, predicted_labels = torch.max(probs, 1)

            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_image_paths.extend(img_paths)
            all_confidences.extend(confidences.cpu().numpy())
            all_similarity_scores_list.append(probs.cpu().numpy())
            all_contexts_text.extend(contexts_text) # These are raw text descriptions

    # Concatenate all similarity scores
    all_similarity_scores = np.vstack(all_similarity_scores_list) if all_similarity_scores_list else np.array([])

    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(len(emotions)))) if all_labels else np.zeros((len(emotions), len(emotions)))
    class_report = classification_report(all_labels, all_preds, target_names=emotions, digits=4, zero_division=0) if all_labels else "No data to report."

    return (
        accuracy, conf_matrix, class_report,
        all_preds, all_labels, all_image_paths,
        all_confidences, all_similarity_scores, all_contexts_text
    )


def display_results_with_contexts(results, emotions=config.EMOTIONS, num_samples_per_class=config.NUM_SAMPLES_TO_DISPLAY_PER_CLASS):
    """Displays comprehensive evaluation results including confusion matrix, report, and sample contexts."""
    (accuracy, conf_matrix, class_report, all_preds, all_labels,
     all_image_paths, all_confidences, _, all_contexts_text) = results # Ignoring all_similarity_scores for this display

    print(f"\n{'='*60}")
    print("ENHANCED CLIP-ADAPTER EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Display confusion matrix as text
    print(f"\n{'='*40}")
    print("CONFUSION MATRIX")
    print(f"{'='*40}")
    conf_df = pd.DataFrame(conf_matrix, index=emotions, columns=emotions)
    print(conf_df)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    # plt.show() # Optionally show plot immediately, or save it
    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix plot saved as confusion_matrix.png")


    # Display classification report
    print(f"\n{'='*40}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*40}")
    print(class_report)

    # Display sample contexts for each emotion class
    print(f"\n{'='*60}")
    print("SAMPLE CONTEXT DESCRIPTIONS BY TRUE EMOTION CLASS")
    print(f"{'='*60}")

    emotion_samples = {i: [] for i in range(len(emotions))}
    for i in range(len(all_labels)):
        true_label = all_labels[i]
        emotion_samples[true_label].append({
            'predicted_label': all_preds[i],
            'confidence': all_confidences[i],
            'image_path': all_image_paths[i],
            'context': all_contexts_text[i],
            'correct': all_labels[i] == all_preds[i]
        })

    for emotion_idx, emotion_name in enumerate(emotions):
        print(f"\n{'-'*50}")
        print(f"EMOTION (TRUE): {emotion_name.upper()}")
        print(f"{'-'*50}")

        samples = emotion_samples[emotion_idx]
        if not samples:
            print("No samples found for this emotion.")
            continue

        samples.sort(key=lambda x: x['confidence'], reverse=True) # Sort by confidence

        correct_samples = [s for s in samples if s['correct']]
        incorrect_samples = [s for s in samples if not s['correct']]

        print(f"Total samples for {emotion_name}: {len(samples)}")
        print(f"Correct predictions: {len(correct_samples)}")
        print(f"Incorrect predictions: {len(incorrect_samples)}")

        if correct_samples:
            print(f"\n  ✅ TOP CORRECT PREDICTIONS (True: {emotion_name}):")
            for i, sample in enumerate(correct_samples[:num_samples_per_class]):
                print(f"\n    Sample {i+1}:")
                print(f"      Image: {os.path.basename(sample['image_path'])}")
                print(f"      Predicted: {emotions[sample['predicted_label']]} (Confidence: {sample['confidence']:.4f})")
                print(f"      Context: {sample['context']}")

        if incorrect_samples:
            print(f"\n  ❌ TOP INCORRECT PREDICTIONS (True: {emotion_name}):")
            # Show fewer incorrect samples, e.g., up to 2 or num_samples_per_class
            for i, sample in enumerate(incorrect_samples[:min(2, num_samples_per_class)]):
                print(f"\n    Sample {i+1}:")
                print(f"      Image: {os.path.basename(sample['image_path'])}")
                print(f"      Predicted: {emotions[sample['predicted_label']]} (Confidence: {sample['confidence']:.4f})")
                print(f"      Context: {sample['context']}")
    print(f"\n{'='*60}")


def analyze_context_quality(all_contexts_text, all_labels, emotions=config.EMOTIONS):
    """Analyzes the quality of generated VLM context descriptions."""
    print(f"\n{'='*60}")
    print("CONTEXT DESCRIPTION ANALYSIS")
    print(f"{'='*60}")

    if not all_contexts_text:
        print("No context descriptions provided for analysis.")
        return

    valid_contexts = [c for c in all_contexts_text if c and c.strip() != "No description available" and c.strip() != "Error loading image"] # Added more checks
    num_total_contexts = len(all_contexts_text)
    num_valid_contexts = len(valid_contexts)

    print(f"Total contexts processed: {num_total_contexts}")
    print(f"Valid contexts generated: {num_valid_contexts}")
    if num_total_contexts > 0:
        success_rate = (num_valid_contexts / num_total_contexts) * 100
        print(f"VLM Context Generation Success Rate: {success_rate:.2f}%")
    else:
        print("VLM Context Generation Success Rate: N/A (no contexts)")


    if valid_contexts:
        avg_length = sum(len(c.split()) for c in valid_contexts) / num_valid_contexts
        print(f"Average valid context length: {avg_length:.1f} words")

        # Find common keywords by emotion (simple approach)
        print("\nCOMMON KEYWORDS IN VALID CONTEXTS BY TRUE EMOTION:")
        # Define common stopwords (extend as needed)
        stopwords = set(['the', 'a', 'is', 'in', 'it', 'of', 'and', 'to', 'this', 'person', 'image',
                         'facial', 'expression', 'face', 'shows', 'appears', 'seems', 'like', 'with', 'their'])

        for emotion_idx, emotion_name in enumerate(emotions):
            # Filter contexts for the current true emotion and ensure they are valid
            emotion_contexts = [
                all_contexts_text[i] for i, label in enumerate(all_labels)
                if label == emotion_idx and all_contexts_text[i] and
                   all_contexts_text[i].strip() != "No description available" and
                   all_contexts_text[i].strip() != "Error loading image"
            ]

            if emotion_contexts:
                words = []
                for context_text in emotion_contexts:
                    # Basic cleaning and tokenization
                    cleaned_words = [
                        word.lower().strip('.,!?";:') for word in context_text.split()
                        if len(word.strip('.,!?";:')) > 3 and word.lower() not in stopwords
                    ]
                    words.extend(cleaned_words)

                if words:
                    common_words = Counter(words).most_common(5)
                    print(f"  {emotion_name.capitalize()}: {[word for word, count in common_words]}")
                else:
                    print(f"  {emotion_name.capitalize()}: No significant keywords found after filtering.")
            else:
                print(f"  {emotion_name.capitalize()}: No valid contexts found for this emotion.")
    print(f"\n{'='*60}")