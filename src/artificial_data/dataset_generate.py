import pandas as pd
import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


datasets = {
    # "dataset1": {
    #     "function": make_classification,
    #     "params": {
    #         "n_samples": 10000,
    #         "n_features": 512,
    #         "n_classes": 10,
    #         "n_informative": 20,
    #         "random_state": 42
    #     }
    # },
    # "dataset2": {
    #     "function": make_classification,
    #     "params": {
    #         "n_samples": 12000,
    #         "n_features": 512,
    #         "n_classes": 5,
    #         "n_informative": 10,
    #         "random_state": 42
    #     }
    # },
    "dataset3": {
        "function": make_classification,
        "params": {
            "n_samples": 8000,
            "n_features": 512,
            "n_classes": 20,
            "n_informative": 10,
            "random_state": 42
        }
    },
}


if __name__ == "__main__":
    os.makedirs('artificial_datasets', exist_ok=True)
    for dataset_name, dataset_info in datasets.items():
        X, y = dataset_info["function"](**dataset_info["params"])
        print(f"Generated {dataset_name} with shape:", X.shape, "and labels:", y.shape)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save train set
        df_train = pd.DataFrame(X_train)
        df_train['target'] = y_train
        df_train.to_csv(f"artificial_datasets/{dataset_name}_train.csv", index=False)
        
        # Save test set
        df_test = pd.DataFrame(X_test)
        df_test['target'] = y_test
        df_test.to_csv(f"artificial_datasets/{dataset_name}_test.csv", index=False)
        
        print(f"Saved train set with shape: {X_train.shape}")
        print(f"Saved test set with shape: {X_test.shape}")
