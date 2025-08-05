import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from artificial_data.model_art import Head, ClassificationModel


# Cache para datasets
_dataset_cache = {}

def eval_single_dataset(backbone, dataset_name, device):
    head = get_head(backbone, dataset_name)
    classification_model = ClassificationModel(backbone, head)
    classification_model.eval()

    # Usa cache para datasets
    if dataset_name not in _dataset_cache:
        _dataset_cache[dataset_name] = get_dataset(dataset_name, batch_size=32, shuffle=False)
    dataset = _dataset_cache[dataset_name]

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in dataset:
            X_val, y_val = batch
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = classification_model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_val.cpu().numpy())
    # Calculate accuracy
    accuracy = (torch.tensor(all_predictions) == torch.tensor(all_targets)).float().mean().item()
    return accuracy



# Cache para heads
_head_cache = {}

def get_head(backbone, dataset_name):
    """Get the head for a specific dataset, usando cache."""
    cache_key = (dataset_name, backbone.get_output_dim())
    if cache_key not in _head_cache:
        dataset_model_ = torch.load(f'artificial_checkpoints/mlp_model_{dataset_name}.pth')
        head = Head(backbone.get_output_dim(), dataset_model_['model_config']['output_dim'])
        head.load_state_dict(dataset_model_['head_state_dict'])
        _head_cache[cache_key] = head
    return _head_cache[cache_key]

def get_dataset(dataset_name, batch_size=32, shuffle=False):
    path = f"artificial_datasets_2/{dataset_name}_test.csv"
    val_data = pd.read_csv(path) 
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values

    x_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return val_loader