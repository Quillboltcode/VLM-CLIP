# config.py

from dataclasses import dataclass


@dataclass
class AdapterConfig:
    use_shared: bool = True
    context_layers: int = 2  # Apply ContextAdapter in first N layers
    textual_layers: int = 2  # Apply TextualAdapter in last N layers
    shared_layer_index: int = 6  # Apply SharedAdapter at this layer index

