# Topological Prototype Selection (TPS) - Bifiltration Implementation

## Overview

This repository contains a Python implementation of the **Topological Prototype Selection (TPS)** algorithm, specifically the bifiltration level set approach described in the preprint "Prototype Selection Using Topological Data Analysis" by Eckert et al. (2025).

TPS is a novel prototype selection method that leverages Topological Data Analysis (TDA) and persistent homology to identify representative subsets from large datasets. The algorithm achieves significant data reduction while maintaining or even improving classification performance.

## Key Features

- **Bifiltration-based prototype selection** using both inter-class and intra-class topology
- **Sequential two-step filtration process**:
  1. Neighbor filtration (inter-class separation)
  2. Radius filtration (intra-class structure)
- **Flexible metric support** (Euclidean, Manhattan, cosine, etc.)
- **Preserves class imbalance** structure in prototype selection
- **Significant data reduction** while maintaining classification performance

## Installation

### Requirements

```bash
pip install numpy
pip install scikit-learn
pip install scipy
pip install ripser
pip install matplotlib
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/tps-bifiltration
cd tps-bifiltration
```

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_classification
from bifiltration_prototype_selector import BifiltrationPrototypeSelector

# Generate sample data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=2,
                          class_sep=1.0, random_state=42)

# Initialize the selector
selector = BifiltrationPrototypeSelector(
    k_neighbors=3,               # k-nearest neighbors for inter-class filtration
    homology_dimension=0,        # H0 for connected components
    min_persistence=0.001,       # Minimum persistence threshold
    neighbor_quantile=0.25,      # Quantile for neighbor filtration
    radius_statistic='mean'      # Statistic for radius filtration
)

# Fit and select prototypes for target class
selector.fit(X, y, target_class=1)

# Get the selected prototypes
prototypes, prototype_indices = selector.get_prototypes(X)
print(f"Selected {len(prototype_indices)} prototypes from {len(X)} samples")

# Visualize results (for 2D data)
selector.plot_prototypes(X, y, target_class=1)
```

## Algorithm Description

### Two-Step Filtration Process

1. **Neighbor Filtration (Inter-class)**
   - Measures separation between target and non-target classes
   - Uses sum of distances to k-nearest other-class neighbors
   - Selects vertices whose persistence lifetime is closest to the specified quantile

2. **Radius Filtration (Intra-class)**
   - Captures internal structure of the target class
   - Uses sum of ALL distances to same-class neighbors
   - Applied only to vertices selected from neighbor filtration (level set approach)
   - Selects vertices with persistence lifetime closest to mean/median

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_neighbors` | int | 5 | Number of nearest other-class neighbors for inter-class filtration |
| `homology_dimension` | int | 0 | Dimension of homology to compute (0 for connected components, 1 for loops) |
| `min_persistence` | float | 0.01 | Minimum persistence threshold for topological features |
| `neighbor_quantile` | float | 0.15 | Quantile for selecting neighbor filtration lifetime (0.0 to 1.0) |
| `radius_statistic` | str | 'mean' | Statistic for radius filtration threshold ('mean' or 'median') |
| `metric` | str/callable | 'euclidean' | Distance metric to use |
| `metric_params` | dict | None | Additional parameters for the metric |

### Hyperparameter Guidelines

Based on extensive experiments in the paper:

- **Use lower values of `k_neighbors`** (1-5) to better capture local boundary geometry
- **Tuning `neighbor_quantile` and `min_persistence`** is more important than `k_neighbors` for dataset reduction
- **Lower `neighbor_quantile` values** (0.05-0.25) provide more reduction with geometric regularization (prototypes closer to boundary)
- **Lower `min_persistence` values** if reduction percentage is a priority

## Performance Results

Real data highlights from preprint:

### Real Datasets
- **Average reduction**: 69.3% (superior to CNN+ENN's 59.2%)
- **Average G-Mean difference**: +0.013 (improved performance by removing noisy observations)
- **Computational efficiency**: Generally 2-3x faster than CNN+ENN on medium/large datasets, but can be slower when using higher homology groups

### Text Classification (Cosine Similarity)
- TPS outperformed AllKNN, Bien-Tibshirani, and K-Means methods
- Both CNN+ENN and TPS improved G-Mean performance from baseline models, but TPS had higher reduction percentages

## Advanced Usage

### Using Different Metrics

```python
# Cosine similarity for text data
selector = BifiltrationPrototypeSelector(
    k_neighbors=3,
    metric='cosine'
)

# Manhattan distance for high-dimensional data
selector = BifiltrationPrototypeSelector(
    k_neighbors=5,
    metric='manhattan'
)

# Custom metric function
def custom_metric(X1, X2):
    # Your custom distance computation
    return distance_matrix

selector = BifiltrationPrototypeSelector(
    k_neighbors=3,
    metric=custom_metric
)
```

### Getting Summary Statistics

```python
# After fitting
stats = selector.get_summary_statistics()
print(f"Number of prototypes: {stats['n_prototypes']}")
print(f"Neighbor vertices: {stats['n_neighbor_vertices']}")
print(f"Final vertices: {stats['n_final_vertices']}")
print(f"Radius threshold: {stats['radius_threshold']:.4f}")
print(f"Neighbor threshold: {stats['neighbor_threshold']:.4f}")
```

### Multi-class Prototype Selection

```python
# Select prototypes for all classes
all_prototypes = []
for class_label in np.unique(y):
    selector.fit(X, y, target_class=class_label)
    prototypes, indices = selector.get_prototypes(X)
    all_prototypes.extend(indices)

all_prototypes = np.unique(all_prototypes)
print(f"Total prototypes across all classes: {len(all_prototypes)}")
```

## Citation (TO BE UPDATED AFTER PEER REVIEW PUBLICATION)

If you use this implementation in your research, please cite:

```bibtex
@article{eckert2025tps,
  title={Prototype Selection Using Topological Data Analysis},
  author={Eckert, Jordan and Ceyhan, Elvan and Schenck, Henry},
  journal={Preprint},
  year={2025},
  institution={Auburn University}
}
```

## Paper Reference (TO BE UPDATED AFTER PEER REVIEW PUBLICATION)

Eckert, J., Ceyhan, E., & Schenck, H. (2025). "Prototype Selection Using Topological Data Analysis (Preprint)". Department of Mathematics & Statistics, Auburn University.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](LICENSE) file for details.
For more information about GPL-3.0, visit: https://www.gnu.org/licenses/gpl-3.0.en.html

## Contact

- Jordan Eckert - jpe0018@auburn.edu

## Acknowledgments

This implementation is based on the theoretical framework presented in the preprint "Prototype Selection Using Topological Data Analysis" by Eckert et al. (2025). The work leverages the Ripser library for persistent homology computations. 
