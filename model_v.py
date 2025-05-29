# models.py
import torch
import torch.nn as nn
from transformers import (
    CLIPProcessor,
    CLIPModel,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import (
    process_vision_info,
)  # Make sure this file/function is available
import config  # Import your config file


# --- Adapter Modules ---
class BaseAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class ContextAdapter(BaseAdapter):
    pass


class VisualAdapter(BaseAdapter):
    pass


class TextAdapter(BaseAdapter):
    pass


# --- VLM Context Extractor ---
class VLMContextExtractor:
    def __init__(self, model_name=config.VLM_MODEL_NAME, device=config.DEVICE):
        self.device = device
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",  # Handles multi-GPU or CPU placement
            quantization_config=quantization_config,
        )
        # Freeze VLM model parameters
        for param in self.vlm_model.parameters():
            param.requires_grad = False

        self.vlm_processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=config.VLM_MIN_PIXELS,
            max_pixels=config.VLM_MAX_PIXELS,
        )

        # CLIP for encoding text descriptions (can be part of EnhancedCLIPAdapter too if preferred)
        self.clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        self.clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        self.clip_model = self.clip_model.to(self.device)

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def extract_context_from_image(self, image_pil):
        """Extract emotion context from the full image"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil},
                        {
                            "type": "text",
                            "text": "Describe the emotion and facial expression of the person in this image in detail. Focus on specific facial features like eyes, eyebrows, mouth, and overall expression.",
                        },
                    ],
                }
            ]
            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(
                messages
            )  # From qwen_vl_utils
            inputs = self.vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # Move inputs to the VLM model's device (if device_map="auto" was used, this might not be strictly necessary
            # but it's good practice if you know the inputs should be on a specific device for generation)
            # inputs = {k: v.to(self.vlm_model.device if hasattr(self.vlm_model, 'device') else self.device) for k, v in inputs.items()}
            inputs = inputs.to(self.device)
            with torch.no_grad():
                generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=200)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.vlm_processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            return (
                output_text[0].strip()
                if output_text and output_text[0].strip()
                else None
            )
        except Exception as e:
            print(f"Error processing image for VLM context: {e}")
            return None

    def encode_context_descriptions(self, description):
        """Encode context description using CLIP text encoder"""
        if not description:
            return None
        try:
            text_inputs = self.clip_processor(
                text=[description], padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)  # Ensure tensors are on the correct device for CLIP model
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
        except Exception as e:
            print(f"Error encoding context with CLIP: {e}")
            return None


# --- Enhanced CLIP Adapter Model ---
class EnhancedCLIPAdapter(nn.Module):
    def __init__(
        self,
        clip_model_name=config.CLIP_MODEL_NAME,
        alpha=config.ALPHA,
        beta=config.BETA,
        gamma=config.GAMMA,
        bottleneck_dim=config.BOTTLENECK_DIM,
        device=config.DEVICE,
        vlm_context_extractor=None,
    ):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Freeze CLIP model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Get feature dimensions
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 224, 224).to(self.device)
            image_features = self.model.get_image_features(pixel_values=dummy_image)
            self.image_feature_dim = image_features.shape[-1]
            self.text_feature_dim = (
                self.image_feature_dim
            )  # CLIP text and image features have the same dim

        # Initialize Adapters
        self.visual_adapter = VisualAdapter(self.image_feature_dim, bottleneck_dim).to(
            self.device
        )
        self.text_adapter = TextAdapter(self.text_feature_dim, bottleneck_dim).to(
            self.device
        )
        self.context_adapter = ContextAdapter(self.text_feature_dim, bottleneck_dim).to(
            self.device
        )  # Assuming context features have same dim as CLIP text

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.vlm_context_extractor = (
            vlm_context_extractor
            if vlm_context_extractor
            else VLMContextExtractor(device=self.device)
        )

        self.original_emotion_text_features = {}
        self.adapted_emotion_embedding_tensor = None
        self.emotion_embedding_tensor = None  # Will hold original embeddings

    def encode_emotion_descriptions(self, emotions=config.EMOTIONS):
        """Encodes predefined emotion descriptions using the CLIP text encoder."""
        self.emotion_descriptions = {
            emotion: [f"A person expressing {emotion}"] for emotion in emotions
        }
        self.original_emotion_text_features = {}
        # self.emotion_text_features_per_description = {} # Not strictly needed if only using mean

        with torch.no_grad():
            for emotion, descriptions in self.emotion_descriptions.items():
                emotion_features_list = []
                for description in descriptions:
                    text_inputs = self.processor(
                        text=[description],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    text_outputs = self.model.get_text_features(**text_inputs)
                    text_features_norm = text_outputs / text_outputs.norm(
                        dim=-1, keepdim=True
                    )
                    emotion_features_list.append(text_features_norm)

                if emotion_features_list:
                    all_desc_features = torch.cat(emotion_features_list, dim=0)
                    self.original_emotion_text_features[emotion] = (
                        all_desc_features.mean(dim=0, keepdim=True)
                    )
                else:
                    # Fallback if a description fails, though unlikely with simple prompts
                    self.original_emotion_text_features[emotion] = torch.zeros(
                        1, self.text_feature_dim
                    ).to(self.device)

            self.emotion_embedding_tensor = torch.cat(
                list(self.original_emotion_text_features.values()), dim=0
            ).to(self.device)
            self.update_emotion_embeddings()  # Initialize adapted embeddings

    def update_emotion_embeddings(self):
        """Updates emotion text embeddings using the text adapter."""
        if self.emotion_embedding_tensor is None:
            print(
                "Warning: Original emotion embeddings not encoded. Call encode_emotion_descriptions first."
            )
            return

        with (
            torch.no_grad()
        ):  # Adapters are trained, but this update uses their current state
            original_text_features = self.emotion_embedding_tensor.clone().detach()
            adapter_output = self.text_adapter(original_text_features)
            adapted_feature = (
                self.beta * adapter_output + (1 - self.beta) * original_text_features
            )
            self.adapted_emotion_embedding_tensor = (
                adapted_feature / adapted_feature.norm(dim=-1, keepdim=True)
            )

    def forward(
        self, pixel_values, context_features=None, use_adapters_for_training=True
    ):
        """
        Forward pass for training (calculates loss) or inference (calculates logits).
        Set use_adapters_for_training to True during training loop for adapter outputs,
        and False if you want to use the adapters in eval mode but not necessarily with autograd.
        """
        # Image Features
        with torch.no_grad():  # Original features are not trained
            original_image_features = self.model.get_image_features(
                pixel_values=pixel_values
            )
            original_image_features = (
                original_image_features
                / original_image_features.norm(dim=-1, keepdim=True)
            )

        adapted_image_features = self.visual_adapter(original_image_features)
        final_image_features = (
            self.alpha * adapted_image_features
            + (1 - self.alpha) * original_image_features
        )
        final_image_features = final_image_features / final_image_features.norm(
            dim=-1, keepdim=True
        )

        # Context Features (if provided)
        if (
            context_features is not None and context_features.nelement() > 0
        ):  # Check if tensor is not empty
            # Ensure context_features are on the correct device and have the right shape
            if context_features.shape[-1] != self.text_feature_dim:
                # This can happen if VLM returns None and a zero tensor was created with a different dim
                print(
                    f"Warning: Context feature dimension mismatch. Expected {self.text_feature_dim}, got {context_features.shape[-1]}. Skipping context."
                )
                combined_features = final_image_features
            else:
                adapted_context_features = self.context_adapter(context_features)
                final_context_features = (
                    self.gamma * adapted_context_features
                    + (1 - self.gamma) * context_features
                )
                final_context_features = (
                    final_context_features
                    / final_context_features.norm(dim=-1, keepdim=True)
                )
                combined_features = (
                    final_image_features + final_context_features
                ) / 2.0  # Average fusion
                combined_features = combined_features / combined_features.norm(
                    dim=-1, keepdim=True
                )
        else:
            combined_features = final_image_features

        # Text Features (Emotion Embeddings)
        if (
            self.training
            or not hasattr(self, "adapted_emotion_embedding_tensor")
            or self.adapted_emotion_embedding_tensor is None
        ):
            # During training, text adapter is also being trained, so calculate dynamically
            # Or if adapted_emotion_embedding_tensor hasn't been pre-calculated for eval
            original_text_features = (
                self.emotion_embedding_tensor.clone().detach()
            )  # Start with original
            adapted_text_features_output = self.text_adapter(original_text_features)
            final_text_features = (
                self.beta * adapted_text_features_output
                + (1 - self.beta) * original_text_features
            )
            final_text_features = final_text_features / final_text_features.norm(
                dim=-1, keepdim=True
            )
        else:
            # For evaluation, use pre-calculated adapted embeddings
            final_text_features = self.adapted_emotion_embedding_tensor

        # Calculate logits
        temperature = self.model.logit_scale.exp()  # CLIP's learned temperature
        logits = temperature * torch.matmul(combined_features, final_text_features.T)
        return logits

    def predict_probs(self, pixel_values, context_features=None):
        """Generates probability scores for each emotion class."""
        self.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            logits = self.forward(
                pixel_values, context_features, use_adapters_for_training=False
            )
            probs = torch.softmax(logits, dim=1)
        return probs

    def get_trainable_parameters(self):
        params = []
        params.extend(list(self.visual_adapter.parameters()))
        params.extend(list(self.text_adapter.parameters()))
        params.extend(list(self.context_adapter.parameters()))
        return params

    def print_model_structure(self):
        print("\n=== Enhanced CLIP-Adapter Model Structure ===")
        total_params = 0

        print("\nCLIP Model (Backbone, Frozen):")
        clip_params = sum(p.numel() for p in self.model.parameters())
        total_params += clip_params
        print(f"CLIP Model Parameters: {clip_params:,}")

        if self.vlm_context_extractor:
            print("\nVLM Context Extractor (Frozen):")
            vlm_params = sum(
                p.numel() for p in self.vlm_context_extractor.vlm_model.parameters()
            )
            total_params += vlm_params
            print(f"VLM Parameters: {vlm_params:,}")
            # Also count params of CLIP used by VLMContextExtractor if it's distinct
            if (
                hasattr(self.vlm_context_extractor, "clip_model")
                and self.vlm_context_extractor.clip_model is not self.model
            ):
                vlm_clip_params = sum(
                    p.numel()
                    for p in self.vlm_context_extractor.clip_model.parameters()
                )
                total_params += vlm_clip_params
                print(f"VLM's CLIP Encoder Parameters: {vlm_clip_params:,}")

        adapter_params_total = 0
        for name, adapter in [
            ("Visual", self.visual_adapter),
            ("Text", self.text_adapter),
            ("Context", self.context_adapter),
        ]:
            print(f"\n{name} Adapter (Trainable):")
            current_adapter_params = 0
            for module_name, module in adapter.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules
                    params = sum(
                        p.numel() for p in module.parameters() if p.requires_grad
                    )
                    if params > 0:
                        print(f"  {module_name}: {params:,} parameters")
                        current_adapter_params += params
            adapter_params_total += current_adapter_params
            print(f"  Total {name} Adapter Parameters: {current_adapter_params:,}")

        total_params += adapter_params_total  # Add trainable adapter params to total
        print(
            f"\nTotal Parameters (Frozen Backbone + Trainable Adapters): {total_params:,}"
        )
        print(f"Trainable Parameters (Adapters Only): {adapter_params_total:,}")

