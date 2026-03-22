# Lab 2: Classification Using KNN and RNN Algorithms

**Course:** Advanced Big Data and Data Mining (MSCS-634-B01)  
**Student:** Rajiv Agrawal
**Due Date:** March 22, 2026
**Instructor** Satish Penmatsa

---

## Purpose

This lab explores two instance-based (lazy learning) classification algorithms — **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** — applied to the **Wine Dataset** from `sklearn`. The goal is to understand how the choice of hyperparameters (k for KNN, radius for RNN) affects model accuracy, and to compare the strengths and weaknesses of both approaches on a real multi-class dataset.

---

## Dataset

- **Source:** `sklearn.datasets.load_wine()`
- **Samples:** 178 wine instances
- **Features:** 13 chemical properties (e.g., alcohol, malic acid, proline)
- **Classes:** 3 wine varieties (class_0, class_1, class_2)
- **Split:** 80% training / 20% testing (stratified, random_state=42)

---

## Key Results and Observations

### KNN – Accuracy vs. k

| k  | Accuracy |
|----|----------|
| 1  | 77.78%   |
| 5  | **80.56%** ← Best |
| 11 | 80.56%   |
| 15 | 80.56%   |
| 21 | 80.56%   |

**Observation:** KNN accuracy improved from 77.78% at k=1 to 80.56% at k=5, then plateaued for all higher k values. k=1 overfits to individual noise points. The accuracy stabilized at 80.56% from k=5 onward, indicating that the Wine Dataset's decision boundaries are well-captured with a small-to-moderate neighborhood.

### RNN – Accuracy vs. Radius

| Radius | Accuracy |
|--------|----------|
| 350    | **72.22%** ← Best |
| 400    | 69.44%   |
| 450    | 69.44%   |
| 500    | 69.44%   |
| 550    | 66.67%   |
| 600    | 66.67%   |

**Observation:** Accuracy **decreased** as radius increased — the opposite of what might be expected. The Wine Dataset was used without feature scaling, so Euclidean distances are dominated by the *proline* feature (range: 278–1680). At larger radii, the classifier pulls in far-away samples from different classes, introducing noise. class_2 was especially affected, with poor recall across all radius values. The smallest radius (350) was optimal as it kept neighborhoods tight and class-homogeneous.

### KNN vs. RNN Comparison

| Aspect | KNN | RNN |
|--------|-----|-----|
| Neighborhood size | Fixed (always k neighbors) | Variable (all within radius) |
| Outlier risk | None — always finds k neighbors | Points outside all neighborhoods need special handling |
| Parameter tuning | Tune integer k | Tune real-valued radius (scale-dependent) |
| Best when | Data density is irregular | Domain-meaningful distance threshold exists |

**KNN is preferable** when data density varies across the feature space, and when a fixed, predictable neighborhood is desired. It is also more robust if no natural distance threshold is known.

**RNN is preferable** when a domain-meaningful spatial threshold exists (e.g., sensor range, geographic proximity), or when local density information is important for classification.

---

## Files

| File | Description |
|------|-------------|
| `Lab2_KNN_RNN_Classification.ipynb` | Main Jupyter Notebook with all code, plots, and discussion |
| `README.md` | This file – summary of the lab |
| `knn_accuracy_vs_k.png` | Plot of KNN accuracy across k values (generated on run) |
| `rnn_accuracy_vs_radius.png` | Plot of RNN accuracy across radius values (generated on run) |
| `knn_vs_rnn_comparison.png` | Side-by-side comparison plot (generated on run) |

---

## Challenges and Decisions

1. **Radius selection for RNN:** The lab specifies radius values of 350–600, which only make sense when the Wine Dataset is used *without feature scaling*. The unscaled *proline* feature dominates the Euclidean distance, making these radius values appropriate. A note is included in the notebook to explain this design decision.

2. **Outlier handling in RNN:** `RadiusNeighborsClassifier` raises an error when a test sample has no neighbors within the radius. The `outlier_label='most_frequent'` parameter was used to assign the majority class label to such points, preventing exceptions and allowing fair accuracy evaluation across all radius values.

3. **Stratified splitting:** A stratified train-test split was used to ensure all three wine classes are proportionally represented in both training and testing sets, avoiding evaluation bias from class imbalance.

---

## How to Run

1. Open `Lab2_KNN_RNN_Classification.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure `scikit-learn`, `pandas`, `numpy`, and `matplotlib` are installed:
   ```bash
   pip install scikit-learn pandas numpy matplotlib
   ```
3. Run all cells from top to bottom. Plots will be displayed inline and saved as PNG files.
# Lab-2
