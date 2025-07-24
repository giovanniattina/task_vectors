# Editing Models with Task Arithmetic

This repository is inspired by the code from the ICLR 2023 paper [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089), by Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi and Ali Farhadi.

### Install dependencies

```bash
conda env create
conda activate task-vectors
```


### Add directory to PYTHONPATH:

```bash
cd task_vectors
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Using task vectors

The task vector logic can be found at [src/task_vectors.py](src/task_vectors.py).

To create a task vector, you will need a pre-trained checkpoint and a fine-tuned checkpoint:

```python
from task_vectors import TaskVector
task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
```

## Artificial Data Pipeline

This repository includes a complete pipeline for generating synthetic datasets and training MLP models to support task vector research. The workflow separates feature extraction and classification components, enabling controlled experiments with task vectors.

### Data Generation

Synthetic classification datasets are generated using `sklearn.datasets.make_classification` with configurable parameters:

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
```

### Model Architecture

The MLP model consists of:
- **Backbone**: 10-layer feature extractor with ReLU activations
- **Head**: 3-layer classification component with configurable output dimensions

```python
class BackBone(torch.nn.Module):
    def __init__(self, input_dim):
        # First block: input_dim -> input_dim (5 layers)
        self.block1 = MLPBlock(input_dim, input_dim, num_layers=5)
        
        # Transition layer: dimensionality reduction
        self.transition = torch.nn.Linear(input_dim, input_dim // 2)
        
        # Second block: reduced_dim -> reduced_dim (5 layers)
        self.block2 = MLPBlock(input_dim // 2, input_dim // 2, num_layers=5)
```

### Training and Fine-tuning Workflow

1. **Base Training**: Train the complete model on `dataset1`
   - Learning rate: 0.001
   - Batch size: 32
   - Epochs: 100
   - Save: Complete model and head weights separately

2. **Fine-tuning**: Adapt the pre-trained model to new datasets
   - Load pre-trained backbone weights
   - Replace classification head for new number of classes
   - Fine-tune with lower learning rate (0.0001)
   - Option to freeze backbone or fine-tune entire model

*For detailed implementation, see the documentation in [src/artificial_data/readme.md](src/artificial_data/readme.md).*

# Artificial Data Experiment

This repository  includes experiments with artificial datasets using MLP models to demonstrate the task vector operations. The experiment compares different arithmetic operations (subtract, add, multiply, divide) applied to task vectors and evaluates their impact on model performance.

```python
import pandas as pd
import matplotlib.pyplot as plt
from task_vectors import TaskVector
from evaluation import eval_single_dataset

# Configuration for artificial data experiment
pretrained_checkpoint = 'artificial_checkpoints/mlp_model.pth'
finetuned_checkpoint = 'artificial_checkpoints/mlp_model_dataset2.pth'
dataset = 'dataset2'
device = 'cpu'

# Test different task vector operations
results = []
for operation in ['subtract', 'add', 'multiply', 'divide']:
    task_vector = TaskVector(
        pretrained_checkpoint=pretrained_checkpoint,
        finetuned_checkpoint=finetuned_checkpoint,
        operation=operation,
    )
    for i in range(10):
        print(f"Iteration {i+1}")
        model_backbone = task_vector.apply_to(pretrained_checkpoint, 0.1*i)
        acc = eval_single_dataset(model_backbone, dataset, device)
        results.append((operation, i, acc))

# Create DataFrame and visualize results
results_df = pd.DataFrame(results, columns=['operation', 'iteration', 'accuracy'])

# Plot results
plt.figure(figsize=(12, 6))
for operation in results_df['operation'].unique():
    subset = results_df[results_df['operation'] == operation]
    plt.plot(subset['iteration'], subset['accuracy'], label=operation)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Task Vector Operations on MLP Model')
plt.legend()
plt.grid()
plt.show()
```

*Note: This experiment uses the basic structure of the task vector code adapted for artificial datasets with MLP models to demonstrate the effects of different arithmetic operations on task vectors.*

