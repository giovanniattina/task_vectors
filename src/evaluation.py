import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from artificial_data.model_art import Head, ClassificationModel

def eval_single_dataset(backbone, dataset_name, device):

  
    head = get_head(backbone, dataset_name)
    classification_model = ClassificationModel(backbone, head)
    classification_model.eval()

    # Load the dataset
    dataset = get_dataset(dataset_name, batch_size=32, shuffle=False)

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in dataset:
            X_val, y_val = batch
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = classification_model(X_val)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_val.cpu().numpy())

    # Calculate accuracy
    accuracy = (torch.tensor(all_predictions) == torch.tensor(all_targets)).float().mean().item()
    return accuracy


def get_head(backbone, dataset_name):
    """Get the head for a specific dataset."""
    dataset_model_ = torch.load(f'artificial_checkpoints/mlp_model_{dataset_name}.pth')
    head = Head(backbone.get_output_dim(), dataset_model_['model_config']['output_dim'])
    head.load_state_dict(dataset_model_['head_state_dict'])
    return head

def get_dataset(dataset_name, batch_size=32, shuffle=False):
    path = f"artificial_datasets/{dataset_name}_test.csv"
    val_data = pd.read_csv(path) 
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values

    x_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return val_loader