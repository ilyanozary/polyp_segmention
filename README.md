# Polyp Segmentation using DeepLabV3+ and MobileNetV2

This project implements a polyp segmentation model using DeepLabV3+ architecture with MobileNetV2 as the backbone. The model is trained on the Kvasir-SEG dataset for medical image segmentation.

## Project Structure

```
polyp_segmention/
├── src/
│   ├── data/
│   │   └── data_loader.py      # Data loading and preprocessing
│   ├── models/
│   │   └── deeplabv3_plus.py   # DeepLabV3+ model implementation
│   ├── metrics/
│   │   └── metrics.py          # Custom metrics and loss functions
│   ├── utils/
│   │   └── visualization.py    # Visualization utilities
│   └── train.py                # Main training script
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Features

- DeepLabV3+ architecture with MobileNetV2 backbone
- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction
- Custom loss functions (Focal Loss + Dice Loss)
- Multiple evaluation metrics (IoU, Dice Coefficient, Precision, Recall)
- Visualization tools for model predictions

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd polyp_segmention
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the following structure:
```
dataset/
├── images/
│   └── *.jpg
└── masks/
    └── *.jpg
```

2. Train the model:
```bash
python src/train.py
```

## Model Architecture

The model consists of:
- MobileNetV2 backbone for feature extraction
- ASPP module for multi-scale feature extraction
- Decoder with skip connections
- Final segmentation head with sigmoid activation

## Training

The model is trained with:
- Mixed loss (Focal Loss + Dice Loss)
- Cosine learning rate decay
- AdamW optimizer
- Batch size of 4
- 10 epochs (configurable)

## Evaluation Metrics

- Intersection over Union (IoU)
- Dice Coefficient
- Precision
- Recall
- Binary Accuracy

## License

[Your License]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_citation,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2023}
}
```

# Files
- train.txt contains the name of the training images and their corresponding masks.
- val.txt contains the name of the validation images and their corresponding masks. 

We have provided train-val split for the fair comparison of the model. 
