---
applyTo: '**/*.py'
---

# CLIP Model Finetuning with Adapters - Coding Standards

## Architecture Overview
- Use CLIP as the base model with adapter-based fine-tuning
- Implement modular adapter components for text and vision
- Support cross-modal attention mechanisms
- Maintain original CLIP architecture while adding adaptation layers

## Code Organization
- Keep adapter implementations in separate modules
- Use clear class hierarchies for different adapter types
- Implement flexible configuration options
- Follow PEP 8 style guidelines

## Model Requirements
- Base CLIP model should remain frozen by default
- Adapters should be easily enabled/disabled
- Support for different adapter architectures:
  - Text-only adapters
  - Vision-only adapters
  - Shared cross-modal adapters
  - Pre-layer norm adapters

## Implementation Standards
1. Type Hints:
   - Use Python type hints for all function arguments and return values
   - Document complex types using docstrings

2. Documentation:
   - Add docstrings to all classes and methods
   - Include parameter descriptions and return value explanations
   - Document shape information for tensor operations

3. Error Handling:
   - Implement proper error checking for model inputs
   - Validate configuration parameters
   - Provide meaningful error messages

4. Performance Considerations:
   - Optimize memory usage in adapter implementations
   - Support batch processing
   - Enable gradient checkpointing when needed

## Testing Requirements
- Unit tests for all adapter components
- Integration tests for the complete model
- Test cases for different adapter configurations
- Validation of saved/loaded adapter weights

## Dependencies
- torch >= 1.10.0
- transformers >= 4.30.0
- numpy >= 1.21.0

## Model Usage Examples
```python
# Example of initializing the model with adapters
model = CLIPWithAdapters(
    clip_model_name="openai/clip-vit-base-patch32",
    text_adapter_size=256,
    vision_adapter_size=256,
    shared_adapter_layers=2,
    use_pre_layer_norm_adapter=True
)

# Example of forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    return_loss=True
)
```

## File Structure
```
project/
├── model.py              # Main model implementation
├── adapter/
│   ├── __init__.py
│   ├── clip_adapter.py   # Adapter implementations
│   └── utils.py         # Utility functions
├── config/
│   └── adapter_config.py # Adapter configurations
└── tests/
    ├── test_model.py
    └── test_adapters.py
```

## Additional Notes
- Keep adapter implementations modular and reusable
- Maintain compatibility with original CLIP functionality
- Support both training and inference modes
- Enable easy experimentation with different adapter architectures