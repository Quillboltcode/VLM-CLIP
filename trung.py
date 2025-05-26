# Define constants
BATCH_SIZE = 32
MODEL_NAME = "openai/clip-vit-large-patch14"
TEST_DIR = "/kaggle/working/RAFDB/test"  # Path to your test folder
TRAIN_DIR = "/kaggle/working/RAFDB/train"  # Path to your train folder
BOTTLENECK_DIM = 64  # Dimensionality of the bottleneck layer
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5
ALPHA = 0.2  # Residual ratio for visual branch (can be tuned)
BETA = 0.2   # Residual ratio for text branch (can be tuned)

# RAFDB emotion labels 
EMOTIONS = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

# Detailed emotion descriptions for zero-shot classification
def get_emotion_descriptions():
    """Create detailed descriptions for each emotion category"""
    
    descriptions = {
        "angry": [
            "the image of an angry facial emotion with furrowed brows and clenched teeth", 
            "a person expressing anger with narrowed eyes and tightened jaw",
            "a face showing intense frustration and hostility",
            "an irritated facial expression with a glaring stare",
            "a person displaying rage with tensed facial muscles"
        ],
        "disgust": [
            "the image of a disgusted facial emotion with wrinkled nose and raised upper lip",
            "a person expressing revulsion with a grimace and squinted eyes",
            "a face showing strong aversion with curled lip",
            "a nauseated facial expression with furrowed brows",
            "a person displaying distaste with pulled back lips"
        ],
        "fear": [
            "the image of a fearful facial emotion with widened eyes and raised eyebrows",
            "a person expressing terror with a dropped jaw and pulled-back lips",
            "a face showing panic with tense mouth and dilated pupils",
            "a frightened facial expression with raised upper eyelids",
            "a person displaying anxiety with frozen stare and pale complexion"
        ],
        "happy": [
            "the image of a happy facial emotion with upturned mouth corners and crinkled eyes",
            "a person expressing joy with a broad smile and relaxed face",
            "a face showing delight with raised cheeks and visible teeth",
            "a cheerful facial expression with beaming smile and bright eyes",
            "a person displaying pleasure with dimples and lifted cheeks"
        ],
        "neutral": [
            "the image of a neutral facial emotion with relaxed features and natural expression",
            "a person with an emotionless face showing no particular feeling",
            "a face with a balanced expression, neither positive nor negative",
            "a composed facial expression with resting features",
            "a person displaying a calm and unemotional demeanor"
        ],
        "sad": [
            "the image of a sad facial emotion with downturned mouth and drooping eyelids",
            "a person expressing sorrow with furrowed brows and quivering lips",
            "a face showing grief with lowered gaze and compressed lips",
            "a melancholic facial expression with sunken cheeks",
            "a person displaying unhappiness with glazed or teary eyes"
        ],
        "surprise": [
            "the image of a surprised facial emotion with raised eyebrows and widened eyes",
            "a person expressing astonishment with an open mouth and stretched skin",
            "a face showing shock with expanded pupils and heightened alertness",
            "a startled facial expression with dropped jaw and gasping mouth",
            "a person displaying amazement with rounded eyes and lifted brows"
        ]
    }
    
    return descriptions

class FolderDataset(Dataset):
    def __init__(self, root_dir, processor, mode='test'):
        """
        Args:
            root_dir: Root directory with emotion subfolders containing images
            processor: CLIP processor for preprocessing images
            mode: 'train' or 'test'
        """
        self.root_dir = root_dir
        self.processor = processor
        self.mode = mode
        self.images = []
        self.labels = []
        self.image_paths = []  # Store paths for visualization

        # Load images and labels from directory structure
        for emotion_idx, emotion_name in enumerate(EMOTIONS):
            emotion_folder = os.path.join(root_dir, emotion_name)
            if not os.path.exists(emotion_folder):
                print(f"Warning: {emotion_folder} does not exist!")
                continue

            for img_file in os.listdir(emotion_folder):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(emotion_folder, img_file)
                    self.images.append(img_path)
                    self.labels.append(emotion_idx)
                    self.image_paths.append(img_path)

        print(f"Loaded {len(self.images)} images for {mode} mode")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # Get label
        emotion_idx = self.labels[idx]

        # Process image for CLIP
        processed_image = self.processor(
            images=image,
            return_tensors="pt",
            padding=True
        )

        return processed_image['pixel_values'][0], torch.tensor(emotion_idx, dtype=torch.long), img_path

class VisualAdapter(nn.Module):
    """Adapter module for the visual branch of CLIP"""
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TextAdapter(nn.Module):
    """Adapter module for the text branch of CLIP"""
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class CLIPAdapter:
    """CLIP-Adapter: Fine-tuning CLIP with bottleneck adapters for few-shot learning"""

    def __init__(self, model_name, alpha=0.2, beta=0.2, bottleneck_dim=64):
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze the CLIP backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Get feature dimensions
        with torch.no_grad():
            # Get a dummy image to determine feature dimensions
            dummy_image = torch.zeros(1, 3, 224, 224).to(device)
            image_features = self.model.get_image_features(pixel_values=dummy_image)
            self.image_feature_dim = image_features.shape[-1]

            # Get text feature dimension (same as image feature dimension in CLIP)
            self.text_feature_dim = self.image_feature_dim

        # Initialize adapters
        self.visual_adapter = VisualAdapter(self.image_feature_dim, bottleneck_dim).to(device)
        self.text_adapter = TextAdapter(self.text_feature_dim, bottleneck_dim).to(device)

        # Set residual ratios
        self.alpha = alpha  # Visual branch residual ratio
        self.beta = beta    # Text branch residual ratio

        # Generate emotion descriptions
        self.emotion_descriptions = get_emotion_descriptions()

        # Pre-encode text descriptions
        self.encode_emotion_descriptions()

    def encode_emotion_descriptions(self):
        """Pre-compute text embeddings for each emotion description"""
        print("Encoding emotion descriptions...")

        self.original_emotion_text_features = {}
        self.emotion_text_features_per_description = {}

        with torch.no_grad():
            for emotion, descriptions in self.emotion_descriptions.items():
                # Store features for each individual description
                self.emotion_text_features_per_description[emotion] = []
                
                # Process each description individually
                for description in descriptions:
                    # Process text with CLIP processor
                    text_inputs = self.processor(
                        text=[description],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)

                    # Get text features
                    text_outputs = self.model.get_text_features(**text_inputs)

                    # Normalize features
                    text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
                    
                    # Store individual description feature
                    self.emotion_text_features_per_description[emotion].append(text_features)

                # Stack all descriptions for this emotion
                all_desc_features = torch.cat(self.emotion_text_features_per_description[emotion], dim=0)
                
                # Store average embedding for this emotion
                self.original_emotion_text_features[emotion] = all_desc_features.mean(dim=0, keepdim=True)

            # Stack all emotion embeddings for batch processing
            self.emotion_embedding_tensor = torch.cat(list(self.original_emotion_text_features.values()), dim=0)

    def update_emotion_embeddings(self):
        """Update the emotion embeddings using the text adapter"""
        with torch.no_grad():
            # Apply text adapter to original emotion text features
            adapted_features = {}
            for emotion, features in self.original_emotion_text_features.items():
                # Apply the adapter
                adapter_output = self.text_adapter(features)

                # Apply residual connection with beta
                adapted_feature = self.beta * adapter_output + (1 - self.beta) * features

                # Normalize the features
                adapted_feature = adapted_feature / adapted_feature.norm(dim=-1, keepdim=True)

                adapted_features[emotion] = adapted_feature

            # Update the emotion embedding tensor for inference
            self.adapted_emotion_embedding_tensor = torch.cat(list(adapted_features.values()), dim=0)

    def train(self, train_loader, num_epochs=50, learning_rate=3e-4):
        """Train the adapter modules"""
        # Set adapters to training mode
        self.visual_adapter.train()
        self.text_adapter.train()

        # Setup optimizer for adapter modules only
        optimizer = optim.Adam(
            list(self.visual_adapter.parameters()) +
            list(self.text_adapter.parameters()),
            lr=learning_rate
        )

        # Temperature parameter from CLIP
        temperature = self.model.logit_scale.exp().item()

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for pixel_values, labels, _ in progress_bar:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

                # Get original CLIP image features
                with torch.no_grad():
                    original_image_features = self.model.get_image_features(pixel_values=pixel_values)
                    original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)

                # Apply visual adapter
                adapted_image_features = self.visual_adapter(original_image_features)

                # Apply residual connection with alpha
                final_image_features = self.alpha * adapted_image_features + (1 - self.alpha) * original_image_features

                # Normalize features
                final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)

                # Get original text features for each class
                original_text_features = self.emotion_embedding_tensor.clone().detach()

                # Apply text adapter
                adapted_text_features = self.text_adapter(original_text_features)

                # Apply residual connection with beta
                final_text_features = self.beta * adapted_text_features + (1 - self.beta) * original_text_features

                # Normalize features
                final_text_features = final_text_features / final_text_features.norm(dim=-1, keepdim=True)

                # Compute logits
                logits = temperature * torch.matmul(final_image_features, final_text_features.T)

                # Compute contrastive loss (cross-entropy)
                loss = nn.CrossEntropyLoss()(logits, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # Update progress bar
                progress_bar.set_postfix({"Loss": f"{total_loss/batch_count:.4f}"})

            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Update emotion embeddings for evaluation
            self.update_emotion_embeddings()

        # Final update of emotion embeddings
        self.update_emotion_embeddings()

        # Set adapters to evaluation mode
        self.visual_adapter.eval()
        self.text_adapter.eval()

    def predict(self, pixel_values):
        """Predict emotion from image pixel values using the adapted model"""
        with torch.no_grad():
            # Get original image features
            original_image_features = self.model.get_image_features(pixel_values=pixel_values)
            original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)

            # Apply visual adapter
            if hasattr(self, 'visual_adapter'):
                adapted_image_features = self.visual_adapter(original_image_features)

                # Apply residual connection with alpha
                final_image_features = self.alpha * adapted_image_features + (1 - self.alpha) * original_image_features

                # Normalize features
                final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)
            else:
                final_image_features = original_image_features

            # Use adapted emotion embeddings if available
            if hasattr(self, 'adapted_emotion_embedding_tensor'):
                similarity = 100 * torch.matmul(final_image_features, self.adapted_emotion_embedding_tensor.transpose(0, 1))
            else:
                similarity = 100 * torch.matmul(final_image_features, self.emotion_embedding_tensor.transpose(0, 1))

            # Get predicted class
            probs = torch.softmax(similarity, dim=1)

        return probs
    
    def predict_with_all_descriptions(self, pixel_values):
        """Predict emotion using all individual descriptions and aggregate results"""
        with torch.no_grad():
            # Get original image features
            original_image_features = self.model.get_image_features(pixel_values=pixel_values)
            original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)

            # Apply visual adapter
            if hasattr(self, 'visual_adapter'):
                adapted_image_features = self.visual_adapter(original_image_features)
                final_image_features = self.alpha * adapted_image_features + (1 - self.alpha) * original_image_features
                final_image_features = final_image_features / final_image_features.norm(dim=-1, keepdim=True)
            else:
                final_image_features = original_image_features
            
            # Calculate similarities for each emotion using all descriptions
            emotion_scores = []
            
            for emotion_idx, emotion in enumerate(EMOTIONS):
                # Get all descriptions for this emotion
                description_features = self.emotion_text_features_per_description[emotion]
                
                # Apply text adapter to each description
                adapted_desc_features = []
                for desc_feature in description_features:
                    # Apply text adapter
                    adapted_feature = self.text_adapter(desc_feature)
                    
                    # Apply residual connection
                    final_feature = self.beta * adapted_feature + (1 - self.beta) * desc_feature
                    
                    # Normalize
                    final_feature = final_feature / final_feature.norm(dim=-1, keepdim=True)
                    adapted_desc_features.append(final_feature)
                
                # Calculate similarity with each description
                desc_similarities = []
                for desc_feature in adapted_desc_features:
                    similarity = 100 * torch.matmul(final_image_features, desc_feature.transpose(0, 1))
                    desc_similarities.append(similarity)
                
                # Stack all description similarities
                all_desc_similarities = torch.cat(desc_similarities, dim=1)
                
                # Take the maximum similarity across all descriptions for this emotion
                max_similarity, _ = torch.max(all_desc_similarities, dim=1)
                emotion_scores.append(max_similarity)
            
            # Stack all emotion scores
            all_emotion_scores = torch.stack(emotion_scores, dim=1)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(all_emotion_scores, dim=1)
            
        return probs

class ZeroShotEmotionRecognition:
    """Zero-shot emotion recognition using CLIP with detailed descriptions"""

    def __init__(self, model_name):
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Generate emotion descriptions
        self.emotion_descriptions = get_emotion_descriptions()

        # Pre-encode text descriptions
        self.encode_emotion_descriptions()

    def encode_emotion_descriptions(self):
        """Pre-compute text embeddings for each emotion description"""
        print("Encoding emotion descriptions...")

        self.emotion_text_features = {}
        self.emotion_text_features_per_description = {}

        with torch.no_grad():
            for emotion, descriptions in self.emotion_descriptions.items():
                # Store features for each individual description
                self.emotion_text_features_per_description[emotion] = []
                
                # Process each description individually
                for description in descriptions:
                    # Process text with CLIP processor
                    text_inputs = self.processor(
                        text=[description],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)

                    # Get text features
                    text_outputs = self.model.get_text_features(**text_inputs)

                    # Normalize features
                    text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
                    
                    # Store individual description feature
                    self.emotion_text_features_per_description[emotion].append(text_features)

                # Stack all descriptions for this emotion
                all_desc_features = torch.cat(self.emotion_text_features_per_description[emotion], dim=0)

                # Store average embedding for this emotion
                self.emotion_text_features[emotion] = all_desc_features.mean(dim=0, keepdim=True)

            # Stack all emotion embeddings for batch processing
            self.emotion_embedding_tensor = torch.cat(list(self.emotion_text_features.values()), dim=0)

    def predict(self, pixel_values):
        """Predict emotion from image pixel values using average of descriptions"""
        with torch.no_grad():
            # Get image features
            image_features = self.model.get_image_features(pixel_values=pixel_values)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity scores with emotion text embeddings (dot product)
            similarity = 100 * torch.matmul(image_features, self.emotion_embedding_tensor.transpose(0, 1))

            # Get predicted class
            probs = torch.softmax(similarity, dim=1)

        return probs
    
    def predict_with_all_descriptions(self, pixel_values):
        """Predict emotion using all individual descriptions and aggregate results"""
        with torch.no_grad():
            # Get image features
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities for each emotion using all descriptions
            emotion_scores = []
            
            for emotion_idx, emotion in enumerate(EMOTIONS):
                # Get all descriptions for this emotion
                description_features = self.emotion_text_features_per_description[emotion]
                
                # Calculate similarity with each description
                desc_similarities = []
                for desc_feature in description_features:
                    similarity = 100 * torch.matmul(image_features, desc_feature.transpose(0, 1))
                    desc_similarities.append(similarity)
                
                # Stack all description similarities
                all_desc_similarities = torch.cat(desc_similarities, dim=1)
                
                # Take the maximum similarity across all descriptions for this emotion
                max_similarity, _ = torch.max(all_desc_similarities, dim=1)
                emotion_scores.append(max_similarity)
            
            # Stack all emotion scores
            all_emotion_scores = torch.stack(emotion_scores, dim=1)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(all_emotion_scores, dim=1)
            
        return probs

def evaluate_model(model, test_loader, use_all_descriptions=False):
    """Evaluate the model on the test set"""

    all_preds = []
    all_labels = []
    all_image_paths = []
    all_confidences = []
    all_similarity_scores = []

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
        digits=4
    )

    all_similarity_scores = np.vstack(all_similarity_scores)

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

    # Calculate average similarity score for each true class
    class_similarities = []
    for class_idx in range(len(EMOTIONS)):
        # Get indices of samples belonging to this class
        class_indices = [i for i, l in enumerate(labels) if l == class_idx]
        if class_indices:
            # Get average similarity scores for this class
            class_similarity = similarity_scores[class_indices].mean(axis=0)
            class_similarities.append(class_similarity)

    # Plot similarities
    plt.figure(figsize=(12, 8))
    x = np.arange(len(EMOTIONS))
    width = 0.1

    for i, class_sim in enumerate(class_similarities):
        plt.bar(x + i*width, class_sim, width, label=f'True: {EMOTIONS[i]}')

    plt.xlabel('Emotion')
    plt.ylabel('Average Similarity Score')
    plt.title(f'{title} Between True Classes and Text Descriptions')
    plt.xticks(x + width * (len(EMOTIONS) - 1) / 2, EMOTIONS)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

def visualize_predictions(image_paths, true_labels, predicted_labels, confidences, num_examples=5, title="Prediction Examples"):
    """Visualize some example predictions"""

    # Choose a mix of correct and incorrect predictions
    correct_indices = [i for i, (y, y_pred) in enumerate(zip(true_labels, predicted_labels)) if y == y_pred]
    incorrect_indices = [i for i, (y, y_pred) in enumerate(zip(true_labels, predicted_labels)) if y != y_pred]

    # Select samples
    num_correct = min(num_examples // 2 + num_examples % 2, len(correct_indices))
    num_incorrect = min(num_examples // 2, len(incorrect_indices))

    if num_correct > 0:
        correct_samples = np.random.choice(correct_indices, num_correct, replace=False)
    else:
        correct_samples = []

    if num_incorrect > 0:
        incorrect_samples = np.random.choice(incorrect_indices, num_incorrect, replace=False)
    else:
        incorrect_samples = []

    selected_indices = list(correct_samples) + list(incorrect_samples)

    # Plot images with predictions
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(15, 4))
    if len(selected_indices) == 1:
        axes = [axes]

    for i, idx in enumerate(selected_indices):
        img_path = image_paths[idx]
        true_label = EMOTIONS[true_labels[idx]]
        pred_label = EMOTIONS[predicted_labels[idx]]
        confidence = confidences[idx]

        # Open and display image
        img = Image.open(img_path).convert('RGB')
        axes[i].imshow(img)

        # Set title with prediction info
        title_text = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}"
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(title_text, color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

def test_single_image(model, image_path, use_all_descriptions=False):
    """Test the model on a single image"""

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    inputs = model.processor(
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Get prediction
    with torch.no_grad():
        if use_all_descriptions and hasattr(model, 'predict_with_all_descriptions'):
            probs = model.predict_with_all_descriptions(inputs.pixel_values)[0]
        else:
            probs = model.predict(inputs.pixel_values)[0]
            
        confidence, predicted = torch.max(probs, 0)

    # Get emotion label
    emotion = EMOTIONS[predicted.item()]

    # Display image with prediction
    plt.figure(figsize=(6, 8))
    plt.imshow(image)
    plt.title(f"Predicted: {emotion} ({confidence.item():.2f})")
    plt.axis('off')
    plt.show()

    # Show emotion probabilities
    plt.figure(figsize=(10, 5))
    sns.barplot(x=EMOTIONS, y=probs.cpu().numpy())
    plt.title('Emotion Similarity Scores')
    plt.ylabel('Similarity Score')
    plt.xlabel('Emotion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print emotion probabilities
    print("\nEmotion similarity scores:")
    for i, emotion_name in enumerate(EMOTIONS):
        print(f"{emotion_name}: {probs[i].item():.4f}")

    # Print text descriptions used for the predicted emotion
    print(f"\nText descriptions used for '{emotion}':")
    for i, desc in enumerate(model.emotion_descriptions[emotion]):
        print(f"{i+1}. {desc}")

    return emotion, confidence.item(), probs.cpu().numpy()

def compare_models(clip_adapter, zero_shot_model, test_loader, use_all_descriptions=True):
    """Compare the performance of CLIP-Adapter and Zero-Shot models with different description strategies"""

    # Evaluate Zero-Shot model with averaged descriptions
    print("\nEvaluating Zero-Shot model with averaged descriptions...")
    zero_shot_results = evaluate_model(zero_shot_model, test_loader, use_all_descriptions=False)
    zero_shot_accuracy = zero_shot_results[0]
    
    # Evaluate Zero-Shot model with all descriptions
    if use_all_descriptions:
        print("\nEvaluating Zero-Shot model with all descriptions...")
        zero_shot_all_desc_results = evaluate_model(zero_shot_model, test_loader, use_all_descriptions=True)
        zero_shot_all_desc_accuracy = zero_shot_all_desc_results[0]
    
    # Evaluate CLIP-Adapter with averaged descriptions
    print("\nEvaluating CLIP-Adapter with averaged descriptions...")
    adapter_results = evaluate_model(clip_adapter, test_loader, use_all_descriptions=False)
    adapter_accuracy = adapter_results[0]
    
    # Evaluate CLIP-Adapter with all descriptions
    if use_all_descriptions:
        print("\nEvaluating CLIP-Adapter with all descriptions...")
        adapter_all_desc_results = evaluate_model(clip_adapter, test_loader, use_all_descriptions=True)
        adapter_all_desc_accuracy = adapter_all_desc_results[0]
    
    # Plot confusion matrices
    plot_confusion_matrix(zero_shot_results[1], "Zero-Shot Confusion Matrix (Avg Descriptions)")
    if use_all_descriptions:
        plot_confusion_matrix(zero_shot_all_desc_results[1], "Zero-Shot Confusion Matrix (All Descriptions)")
    plot_confusion_matrix(adapter_results[1], "CLIP-Adapter Confusion Matrix (Avg Descriptions)")
    if use_all_descriptions:
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
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize zero-shot emotion recognition model for comparison
    print(f"Initializing zero-shot emotion recognition with {MODEL_NAME}...")
    zero_shot_model = ZeroShotEmotionRecognition(MODEL_NAME)

    # Initialize CLIP-Adapter model
    print(f"Initializing CLIP-Adapter with {MODEL_NAME}...")
    clip_adapter = CLIPAdapter(
        MODEL_NAME,
        alpha=ALPHA,
        beta=BETA,
        bottleneck_dim=BOTTLENECK_DIM
    )

    # Create datasets
    train_dataset = FolderDataset(TRAIN_DIR, clip_adapter.processor, mode='train')
    test_dataset = FolderDataset(TEST_DIR, clip_adapter.processor, mode='test')

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # Train CLIP-Adapter
    print("Training CLIP-Adapter...")
    clip_adapter.train(train_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    # Compare models with and without all descriptions
    results = compare_models(clip_adapter, zero_shot_model, test_loader, use_all_descriptions=True)

    # Option to test on single image
    test_image = input("Do you want to test on a single image? (y/n): ")
    if test_image.lower() == 'y':
        image_path = input("Enter path to image: ")
        if os.path.exists(image_path):
            print("\n=== Zero-Shot Results (Averaged Descriptions) ===")
            test_single_image(zero_shot_model, image_path, use_all_descriptions=False)
            
            print("\n=== Zero-Shot Results (All Descriptions) ===")
            test_single_image(zero_shot_model, image_path, use_all_descriptions=True)

            print("\n=== CLIP-Adapter Results (Averaged Descriptions) ===")
            test_single_image(clip_adapter, image_path, use_all_descriptions=False)
            
            print("\n=== CLIP-Adapter Results (All Descriptions) ===")
            test_single_image(clip_adapter, image_path, use_all_descriptions=True)
        else:
            print(f"Error: Image file {image_path} not found.")

if __name__ == "__main__":
    # Ensure we have the necessary imports
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    from tqdm import tqdm
    from torch import nn, optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import CLIPProcessor, CLIPModel
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    # Set global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run the main function
    main()