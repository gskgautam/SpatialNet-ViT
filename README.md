# SpatialNet-ViT: Multimodal Remote Sensing with Vision Transformer + Multi-Task Learning

This repository implements the SpatialNet-ViT model for multimodal remote sensing, following the methodology from the paper:

> "How Can Multimodal Remote Sensing Datasets Transform Classification via SpatialNet-ViT?"

## Project Structure

```
SpatialNet-ViT/
├── data/                # Dataset preparation scripts
├── datasets/            # PyTorch Dataset classes
├── models/              # Model modules (ViT, heads, full model)
├── train/               # Training loop and metrics
├── utils/               # Utility functions (answer mapping, etc.)
├── requirements.txt     # Python dependencies
├── README.md            # Project overview and instructions
```

## Setup

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare datasets**:
   - Run `python data/prepare_ucm.py` for UCM-Caption
   - Run `python data/prepare_rsvqa.py` for RSVQA-LR

## Training

- Edit `train/train.py` as needed for your experiment.
- Run the training script:
  ```bash
  python train/train.py
  ```

## Evaluation

- Metrics (BLEU, METEOR, ROUGE, accuracy, MAE) are implemented in `train/metrics.py`.
- Add or extend evaluation logic in `train/train.py` as needed.

## Research Context

- Implements the full pipeline for:
  - Land-use captioning
  - Visual Question Answering (VQA)
  - Counting
- Modular, extensible, and ready for ablation or extension.

## Citation
If you use this code, please cite the original paper and this repository. 