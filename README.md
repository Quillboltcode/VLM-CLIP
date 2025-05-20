# CLIP Fine-tuning with Adapters

<!--toc:start-->
- [CLIP Fine-tuning with Adapters](#clip-fine-tuning-with-adapters)
  - [📁 Project Structure](#📁-project-structure)
<!--toc:end-->

This repository contains code for fine-tuning CLIP  models using Adapters , allowing efficient parameter-efficient transfer learning. It also supports full fine-tuning by disabling adapters for comparison purposes. The architecture is modular and clean, making it easy to extend or customize for different datasets and tasks.

---

## 📁 Project Structure

```
clip_adapter_finetune/
├── Adapter/               # Contains adapter layers and utilities
│   ├── __init__.py
│   └── clip_adapters.py   # Implementation of adapter modules for CLIP
├── gen_label/             # Scripts to generate augmented labels using LLMs
│   └── label_generator.py
├── model.py               # Main CLIP model with optional adapter support
├── optim_factory.py       # Optimizer and scheduler factory
├── main.py                # Entry point with argument parsing
├── train.py               # Training loop (supports Apex and LoRA)
├── eval.py                # Evaluation script on validation/test set
└── README.md              # This file
```
