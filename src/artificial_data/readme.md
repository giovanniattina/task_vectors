# Artificial Data Experiments

This folder contains a complete pipeline for training and fine-tuning MLP models on synthetic classification datasets. The workflow is designed to support task vector research by separating feature extraction and classification components.

## 📁 Folder Structure

```
artificial_data/
├── README.md                    # This file
├── dataset_generate.py          # Script to generate synthetic datasets
├── model_art.py                 # MLP model definition
├── train_mlp.ipynb             # Training notebook for initial model
├── finetune_mlp.ipynb          # Fine-tuning notebook for new tasks
├── artificial_datasets/         # Generated datasets
│   ├── dataset1_train.csv
│   ├── dataset1_test.csv
│   ├── dataset2_train.csv
│   ├── dataset2_test.csv
│   ├── dataset3_train.csv
│   └── dataset3_test.csv
└── artificial_checkpoints/      # Saved model weights
    ├── mlp_dataset1_model.pth   # Complete model weights
    ├── mlp_dataset1_head.pth    # Head-only weights
    ├── mlp_dataset2_finetuned_model.pth
    ├── mlp_dataset2_finetuned_head.pth
    └── ...
```

## 🔄 Complete Workflow

### Step 1: Generate Classification Datasets

Create synthetic datasets using `sklearn.datasets.make_classification`:

```python
from sklearn.datasets import make_classification
import pandas as pd

# Example configuration for different datasets
datasets = {
    "dataset1": {
        "n_samples": 10000,
        "n_features": 512,
        "n_classes": 10,
        "n_informative": 50,
        "random_state": 42
    },
    "dataset2": {
        "n_samples": 8000,
        "n_features": 512,
        "n_classes": 15,
        "n_informative": 60,
        "random_state": 123
    }
}

# Generate and save datasets
for name, params in datasets.items():
    X, y = make_classification(**params)
    # Split and save train/test CSV files
```

**Key Features:**
- Configurable number of classes, features, and samples
- Consistent feature dimensionality across datasets
- Different classification tasks for transfer learning experiments

### Step 2: Train Base Model (`train_mlp.ipynb`)

Train the initial MLP model on `dataset1`:

**Process:**
1. **Load Data**: Read `dataset1_train.csv` and `dataset1_test.csv`
2. **Model Architecture**: 10-layer MLP with configurable head
3. **Training**: Full model training with configurable parameters
4. **Evaluation**: Comprehensive validation with metrics and visualizations
5. **Save Weights**: 
   - `mlp_dataset1_model.pth` - Complete model state dict
   - `mlp_dataset1_head.pth` - Classification head only

**Key Parameters:**
- `LEARNING_RATE = 0.001`
- `BATCH_SIZE = 32`
- `NUM_EPOCHS = 100`

**Outputs:**
- Trained model weights
- Training/validation curves
- Classification reports and confusion matrices
- Model performance metrics

### Step 3: Fine-tune Model (`finetune_mlp.ipynb`)

Fine-tune the pre-trained model on new datasets:

**Process:**
1. **Load Pre-trained Model**: Load weights from `mlp_dataset1_model.pth`
2. **Dataset Selection**: Choose target dataset (`dataset2` or `dataset3`)
3. **Head Replacement**: Replace final layer to match new number of classes
4. **Transfer Learning**: 
   - Option A: Freeze backbone, train only head
   - Option B: Fine-tune entire model with lower learning rate
5. **Evaluation**: Assess performance on new task
6. **Save Fine-tuned Weights**:
   - `mlp_dataset2_finetuned_model.pth` - Complete fine-tuned model
   - `mlp_dataset2_finetuned_head.pth` - New head weights

**Key Parameters:**
- `FINETUNE_LEARNING_RATE = 0.0001` (Lower than base training)
- `FINETUNE_BATCH_SIZE = 32`
- `FINETUNE_EPOCHS = 50`
- `FREEZE_BACKBONE = False` (Configurable)

## 🏗️ Model Architecture (`model_art.py`)

```python
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        # 10-layer backbone with ReLU activations
        self.layers = ModuleList([Linear + ReLU] * 10)
        
        # 3-layer classification head
        self.head = Sequential(
            Linear(input_dim, input_dim // 2),
            ReLU(),
            Linear(input_dim // 2, output_dim)
        )
```


