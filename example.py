import os
import torch
from model import CLIPWithAdapters
from dataset import create_dataloaders
from trainer import CLIPAdapterTrainer
from glob import glob

def main():
    # Configuration
    clip_model_name = "openai/clip-vit-base-patch32"
    text_adapter_size = 256
    vision_adapter_size = 256
    shared_adapter_layers = 2
    freeze_clip = True
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = CLIPWithAdapters(
        clip_model_name=clip_model_name,
        text_adapter_size=text_adapter_size,
        vision_adapter_size=vision_adapter_size,
        shared_adapter_layers=shared_adapter_layers,
        freeze_clip=freeze_clip
    )
    model.to(device)
    
    # Print model information
    adapter_params = sum(p.numel() for name, p in model.named_parameters() 
                          if 'adapter' in name or 'shared_adapters' in name)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable adapter parameters: {adapter_params:,} ({adapter_params/total_params:.2%} of total)")
    
    # Example data - replace with your actual data paths
    data_dir = "your_data_directory"
    image_paths = sorted(glob(os.path.join(data_dir, "images/*.jpg")))
    
    # Load captions (this is just an example - you'd load your actual captions)
    captions = []
    with open(os.path.join(data_dir, "captions.txt"), "r") as f:
        for line in f:
            captions.append(line.strip())
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        image_paths=image_paths,
        captions=captions,
        processor=model.processor,
        train_val_split=0.9,
        batch_size=32
    )
    
    # Training configuration
    training_args = {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "output_dir": "./clip_adapter_checkpoints"
    }
    
    # Initialize trainer
    trainer = CLIPAdapterTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **training_args
    )
    
    # Train the model
    trainer.train(num_epochs=5, save_every=1, eval_every=1)
    
    # Load best model for inference
    model.load_adapter_weights("./clip_adapter_checkpoints/best_adapter.pt")
    
    # Example of inference
    def encode_text(text):
        inputs = model.processor(text=text, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
        
        return outputs.cpu().numpy()
    
    def encode_image(image_path):
        image = model.processor.image_processor.preprocess(image_path, return_tensors="pt")["pixel_values"]
        image = image.to(device)
        
        with torch.no_grad():
            outputs = model.get_image_features(pixel_values=image)
        
        return outputs.cpu().numpy()
    
    # Example of usage for retrieval
    query_text = "a photo of a dog"
    text_embedding = encode_text(query_text)
    
    # For each image in a set, compute similarity with the query
    for image_path in image_paths[:5]:  # Just check a few as an example
        image_embedding = encode_image(image_path)
        
        # Compute cosine similarity
        similarity = (text_embedding @ image_embedding.T)[0][0]
        print(f"Similarity between '{query_text}' and {image_path}: {similarity:.4f}")

if __name__ == "__main__":
    main()