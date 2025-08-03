import pandas as pd
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def generate_base_dataset(n_samples=15000, n_features=512, n_classes=10, random_state=42):
    """Generate base dataset for pretraining"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=int(n_features * 0.7),  # 70% informative features
        n_redundant=int(n_features * 0.1),    # 10% redundant features
        n_clusters_per_class=2,
        class_sep=1.2,
        random_state=random_state
    )
    return X, y


def generate_related_dataset(base_X, base_y, transformation_type="rotation", noise_level=0.1, 
                           class_mapping=None, random_state=42):
    """Generate a related dataset by transforming the base dataset"""
    np.random.seed(random_state)
    X_new = base_X.copy()
    
    if transformation_type == "rotation":
        # Apply random rotation to feature space
        random_matrix = np.random.normal(0, 1, (base_X.shape[1], base_X.shape[1]))
        rotation_matrix, _ = np.linalg.qr(random_matrix)
        X_new = X_new @ rotation_matrix
        
    elif transformation_type == "feature_shift":
        # Shift importance to different features
        feature_weights = np.random.normal(1.0, 0.3, base_X.shape[1])
        X_new = X_new * feature_weights
        
    elif transformation_type == "nonlinear":
        # Apply nonlinear transformation to subset of features
        n_transform = base_X.shape[1] // 3
        transform_indices = np.random.choice(base_X.shape[1], n_transform, replace=False)
        X_new[:, transform_indices] = np.tanh(X_new[:, transform_indices])
        
    elif transformation_type == "domain_shift":
        # Add domain-specific bias
        domain_bias = np.random.normal(0, 0.5, base_X.shape[1])
        X_new = X_new + domain_bias

    elif transformation_type == "feature_scaling":
        # Apply different scaling to different feature groups
        n_features = base_X.shape[1]
        group_size = n_features // 4
        
        # Create different scaling factors for each group
        scaling_factors = np.ones(n_features)
        scaling_factors[:group_size] *= 2.0  # Amplify first group
        scaling_factors[group_size:2*group_size] *= 0.5  # Reduce second group
        scaling_factors[2*group_size:3*group_size] *= 1.5  # Moderate amplify third group
        # Fourth group remains unchanged (factor = 1.0)
        
        X_new = X_new * scaling_factors
        
        # Add selective noise only to certain features
        noise_features = np.random.choice(n_features, n_features//3, replace=False)
        selective_noise = np.zeros_like(X_new)
        selective_noise[:, noise_features] = np.random.normal(0, noise_level*2, 
                                                             (X_new.shape[0], len(noise_features)))
        X_new = X_new + selective_noise
    
    # Add controlled noise (only if not feature_scaling, which handles noise internally)
    if transformation_type != "feature_scaling":
        noise = np.random.normal(0, noise_level, X_new.shape)
        X_new = X_new + noise
    
    # Add controlled noise
    noise = np.random.normal(0, noise_level, X_new.shape)
    X_new = X_new + noise
    
    # Apply class mapping if provided
    y_new = base_y.copy()
    if class_mapping is not None:
        y_new = np.array([class_mapping.get(label, label) for label in base_y])
    
    return X_new, y_new


def generate_subset_classes(X, y, selected_classes, random_state=42):
    """Generate dataset with subset of classes"""
    np.random.seed(random_state)
    mask = np.isin(y, selected_classes)
    X_subset = X[mask]
    y_subset = y[mask]
    
    # Remap labels to be continuous
    unique_labels = np.unique(y_subset)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y_subset = np.array([label_mapping[label] for label in y_subset])
    
    return X_subset, y_subset


if __name__ == "__main__":
    os.makedirs('artificial_datasets', exist_ok=True)
    
    # Generate base dataset for pretraining
    print("Generating base dataset...")
    X_base, y_base = generate_base_dataset(
        n_samples=15000, 
        n_features=512, 
        n_classes=10, 
        random_state=42
    )
    
    # Save base dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42, stratify=y_base
    )
    
    df_train = pd.DataFrame(X_train)
    df_train['target'] = y_train
    df_train.to_csv("artificial_datasets/base_dataset_train.csv", index=False)
    
    df_test = pd.DataFrame(X_test)
    df_test['target'] = y_test
    df_test.to_csv("artificial_datasets/base_dataset_test.csv", index=False)
    print(f"Base dataset - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Generate Task 1: Rotated feature space (same classes)
    print("\nGenerating Task 1 dataset (rotation transformation)...")
    X_task1, y_task1 = generate_related_dataset(
        X_base, y_base, 
        transformation_type="rotation", 
        noise_level=0.05,
        random_state=123
    )
    
    X_train_t1, X_test_t1, y_train_t1, y_test_t1 = train_test_split(
        X_task1, y_task1, test_size=0.2, random_state=42, stratify=y_task1
    )
    
    df_train_t1 = pd.DataFrame(X_train_t1)
    df_train_t1['target'] = y_train_t1
    df_train_t1.to_csv("artificial_datasets/task1_dataset_train.csv", index=False)
    
    df_test_t1 = pd.DataFrame(X_test_t1)
    df_test_t1['target'] = y_test_t1
    df_test_t1.to_csv("artificial_datasets/task1_dataset_test.csv", index=False)
    print(f"Task 1 dataset - Train: {X_train_t1.shape}, Test: {X_test_t1.shape}")
    
    # Generate Task 2: Subset of classes with domain shift
    print("\nGenerating Task 2 dataset (subset classes + domain shift)...")
    selected_classes = [0, 1, 2, 3, 4, 5]  # 6 out of 10 classes
    X_subset, y_subset = generate_subset_classes(X_base, y_base, selected_classes, random_state=42)
    
    X_task2, y_task2 = generate_related_dataset(
        X_subset, y_subset,
        transformation_type="domain_shift",
        noise_level=0.08,
        random_state=456
    )
    
    X_train_t2, X_test_t2, y_train_t2, y_test_t2 = train_test_split(
        X_task2, y_task2, test_size=0.2, random_state=42, stratify=y_task2
    )
    
    df_train_t2 = pd.DataFrame(X_train_t2)
    df_train_t2['target'] = y_train_t2
    df_train_t2.to_csv("artificial_datasets/task2_dataset_train.csv", index=False)
    
    df_test_t2 = pd.DataFrame(X_test_t2)
    df_test_t2['target'] = y_test_t2
    df_test_t2.to_csv("artificial_datasets/task2_dataset_test.csv", index=False)
    print(f"Task 2 dataset - Train: {X_train_t2.shape}, Test: {X_test_t2.shape}")
    
    # Generate Task 3: Different feature importance with class remapping
    print("\nGenerating Task 3 dataset (feature shift + class remapping)...")
    class_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:0, 6:1, 7:2, 8:3, 9:4}  # 10 -> 5 classes
    X_task3, y_task3 = generate_related_dataset(
        X_base, y_base,
        transformation_type="feature_shift",
        noise_level=0.06,
        class_mapping=class_mapping,
        random_state=789
    )
    
    X_train_t3, X_test_t3, y_train_t3, y_test_t3 = train_test_split(
        X_task3, y_task3, test_size=0.2, random_state=42, stratify=y_task3
    )
    
    df_train_t3 = pd.DataFrame(X_train_t3)
    df_train_t3['target'] = y_train_t3
    df_train_t3.to_csv("artificial_datasets/task3_dataset_train.csv", index=False)
    
    df_test_t3 = pd.DataFrame(X_test_t3)
    df_test_t3['target'] = y_test_t3
    df_test_t3.to_csv("artificial_datasets/task3_dataset_test.csv", index=False)
    print(f"Task 3 dataset - Train: {X_train_t3.shape}, Test: {X_test_t3.shape}")
    
     # Generate Task 4: Feature scaling with selective noise
    print("\nGenerating Task 4 dataset (feature scaling + selective noise)...")
    X_task4, y_task4 = generate_related_dataset(
        X_base, y_base,
        transformation_type="feature_scaling",
        noise_level=0.07,
        random_state=999
    )
    
    X_train_t4, X_test_t4, y_train_t4, y_test_t4 = train_test_split(
        X_task4, y_task4, test_size=0.2, random_state=42, stratify=y_task4
    )
    
    df_train_t4 = pd.DataFrame(X_train_t4)
    df_train_t4['target'] = y_train_t4
    df_train_t4.to_csv("artificial_datasets/task4_dataset_train.csv", index=False)
    
    df_test_t4 = pd.DataFrame(X_test_t4)
    df_test_t4['target'] = y_test_t4
    df_test_t4.to_csv("artificial_datasets/task4_dataset_test.csv", index=False)
    print(f"Task 4 dataset - Train: {X_train_t4.shape}, Test: {X_test_t4.shape}")
    
    print("\nDataset generation complete!")
    print("Base dataset: 10 classes, for pretraining")
    print("Task 1: Same 10 classes, rotated feature space")
    print("Task 2: 6 classes subset, domain shifted")
    print("Task 3: 5 classes (remapped), shifted feature importance")
    print("Task 4: 10 classes, feature scaling with selective noise")