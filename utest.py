import torch

from model import CLIPWithAdapters  # assuming your model file is named model.py

model = CLIPWithAdapters(
    clip_model_name="openai/clip-vit-base-patch32",
    text_adapter_size=256,
    vision_adapter_size=256,
    shared_adapter_layers=2,
    freeze_clip=True,
    use_text_adapter=True,
    use_vision_adapter=True,
    use_shared_adapters=True,
)

# 2. Put model in eval mode
model.eval()

# 3. Load processor
processor = model.processor  # comes from CLIPWithAdapters

# 4. Prepare dummy input
from PIL import Image

image = Image.new("RGB", (224, 224), color="red")  # dummy red image
text = "A red square"


print(type(processor))
print(hasattr(processor, "__call__"))

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
    padding=True,
)

# 5. Forward pass
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        return_loss=True,
    )

# 6. Inspect outputs
print("Loss:", outputs["loss"].item())
print("Text features shape:", outputs["text_features"].shape)
print("Image features shape:", outputs["image_features"].shape)
