"""
Example usage of the Bifiltration Prototype Selector (TPS)

This script demonstrates how to use the TPS algorithm for prototype selection
on various types of datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from bifiltration_prototype_selector import BifiltrationPrototypeSelector


def example_1_basic_usage():
    """Basic usage example with synthetic data."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage with Synthetic Data")
    print("=" * 70)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Original training set size: {len(X_train)}")
    
    # Initialize TPS selector
    selector = BifiltrationPrototypeSelector(
        k_neighbors=3,
        homology_dimension=0,
        min_persistence=0.001,
        neighbor_quantile=0.15,
        radius_statistic='mean'
    )
    
    # Select prototypes for each class
    all_prototype_indices = []
    for class_label in np.unique(y_train):
        print(f"\nSelecting prototypes for class {class_label}...")
        selector.fit(X_train, y_train, target_class=class_label)
        _, indices = selector.get_prototypes(X_train)
        all_prototype_indices.extend(indices)
        print(f"  Selected {len(indices)} prototypes")
    
    # Get unique prototype indices
    prototype_indices = np.unique(all_prototype_indices)
    X_prototypes = X_train[prototype_indices]
    y_prototypes = y_train[prototype_indices]
    
    reduction_pct = (1 - len(prototype_indices) / len(X_train)) * 100
    print(f"\nTotal prototypes selected: {len(prototype_indices)}")
    print(f"Reduction percentage: {reduction_pct:.1f}%")
    
    # Compare classification performance
    print("\n" + "=" * 50)
    print("Classification Performance Comparison")
    print("=" * 50)
    
    # Train on full dataset
    knn_full = KNeighborsClassifier(n_neighbors=3)
    knn_full.fit(X_train, y_train)
    y_pred_full = knn_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)
    
    # Train on prototypes only
    knn_proto = KNeighborsClassifier(n_neighbors=3)
    knn_proto.fit(X_prototypes, y_prototypes)
    y_pred_proto = knn_proto.predict(X_test)
    acc_proto = accuracy_score(y_test, y_pred_proto)
    
    print(f"Accuracy with full training set: {acc_full:.3f}")
    print(f"Accuracy with prototypes only: {acc_proto:.3f}")
    print(f"Accuracy difference: {(acc_proto - acc_full):.3f}")
    
    # Visualize if 2D
    if X.shape[1] == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot full training set
        axes[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
                       c='blue', alpha=0.5, label='Class 0')
        axes[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
                       c='red', alpha=0.5, label='Class 1')
        axes[0].set_title(f'Full Training Set (n={len(X_train)})')
        axes[0].legend()
        
        # Plot prototypes
        axes[1].scatter(X_prototypes[y_prototypes==0, 0], X_prototypes[y_prototypes==0, 1], 
                       c='blue', s=100, marker='*', label='Class 0 prototypes')
        axes[1].scatter(X_prototypes[y_prototypes==1, 0], X_prototypes[y_prototypes==1, 1], 
                       c='red', s=100, marker='*', label='Class 1 prototypes')
        axes[1].set_title(f'Selected Prototypes (n={len(X_prototypes)}, {reduction_pct:.1f}% reduction)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()


def example_2_hyperparameter_comparison():
    """Compare different hyperparameter settings."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Hyperparameter Comparison")
    print("=" * 70)
    
    # Generate dataset
    X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Different configurations to test
    configs = [
        {"name": "Conservative", "neighbor_quantile": 0.5, "k_neighbors": 10},
        {"name": "Balanced", "neighbor_quantile": 0.25, "k_neighbors": 5},
        {"name": "Aggressive", "neighbor_quantile": 0.1, "k_neighbors": 3},
    ]
    
    results = []
    
    for config in configs:
        selector = BifiltrationPrototypeSelector(
            k_neighbors=config["k_neighbors"],
            homology_dimension=0,
            min_persistence=0.001,
            neighbor_quantile=config["neighbor_quantile"],
            radius_statistic='mean'
        )
        
        # Select prototypes for all classes
        all_indices = []
        for class_label in np.unique(y_train):
            selector.fit(X_train, y_train, target_class=class_label)
            _, indices = selector.get_prototypes(X_train)
            all_indices.extend(indices)
        
        prototype_indices = np.unique(all_indices)
        X_proto = X_train[prototype_indices]
        y_proto = y_train[prototype_indices]
        
        # Evaluate
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_proto, y_proto)
        acc = accuracy_score(y_test, knn.predict(X_test))
        
        reduction = (1 - len(prototype_indices) / len(X_train)) * 100
        
        results.append({
            "name": config["name"],
            "n_prototypes": len(prototype_indices),
            "reduction": reduction,
            "accuracy": acc,
            "config": config
        })
        
        print(f"\n{config['name']} Configuration:")
        print(f"  Prototypes: {len(prototype_indices)}")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"  Accuracy: {acc:.3f}")


def example_3_different_metrics():
    """Example using different distance metrics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Different Distance Metrics")
    print("=" * 70)
    
    # Generate high-dimensional data
    X, y = make_classification(
        n_samples=500,
        n_features=20,  # Higher dimensions
        n_redundant=5,
        n_informative=15,
        n_clusters_per_class=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    metrics_to_test = ['euclidean', 'manhattan', 'chebyshev']
    
    for metric_name in metrics_to_test:
        print(f"\n{metric_name.capitalize()} metric:")
        
        selector = BifiltrationPrototypeSelector(
            k_neighbors=5,
            homology_dimension=0,
            min_persistence=0.001,
            neighbor_quantile=0.2,
            radius_statistic='mean',
            metric=metric_name
        )
        
        # Select prototypes
        all_indices = []
        for class_label in np.unique(y_train):
            selector.fit(X_train, y_train, target_class=class_label)
            _, indices = selector.get_prototypes(X_train)
            all_indices.extend(indices)
        
        prototype_indices = np.unique(all_indices)
        reduction = (1 - len(prototype_indices) / len(X_train)) * 100
        
        print(f"  Prototypes selected: {len(prototype_indices)}")
        print(f"  Reduction: {reduction:.1f}%")


def example_4_imbalanced_data():
    """Example with imbalanced dataset."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Imbalanced Dataset")
    print("=" * 70)
    
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=2,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    class_counts = np.bincount(y_train)
    print(f"Original class distribution: {class_counts} (ratio {class_counts[0]/class_counts[1]:.1f}:1)")
    
    selector = BifiltrationPrototypeSelector(
        k_neighbors=3,
        homology_dimension=0,
        min_persistence=0.001,
        neighbor_quantile=0.2,
        radius_statistic='mean'
    )
    
    # Select prototypes for each class
    class_prototypes = {}
    for class_label in np.unique(y_train):
        selector.fit(X_train, y_train, target_class=class_label)
        _, indices = selector.get_prototypes(X_train)
        class_prototypes[class_label] = indices
        print(f"\nClass {class_label}:")
        print(f"  Original samples: {class_counts[class_label]}")
        print(f"  Prototypes selected: {len(indices)}")
        print(f"  Reduction: {(1 - len(indices)/class_counts[class_label])*100:.1f}%")
    
    # Combine all prototypes
    all_indices = np.concatenate(list(class_prototypes.values()))
    all_indices = np.unique(all_indices)
    
    # Check if class imbalance is preserved
    y_prototypes = y_train[all_indices]
    proto_counts = np.bincount(y_prototypes)
    print(f"\nPrototype class distribution: {proto_counts} (ratio {proto_counts[0]/proto_counts[1]:.1f}:1)")
    print("Class imbalance structure preserved: ", 
          "Yes" if (proto_counts[0]/proto_counts[1]) > 2 else "No")


if __name__ == "__main__":
    # Run examples
    example_1_basic_usage()
    example_2_hyperparameter_comparison()
    example_3_different_metrics()
    example_4_imbalanced_data()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
