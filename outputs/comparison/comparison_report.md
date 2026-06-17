# Model Comparison Report: ResNet18 vs ViT-B/16

## 1. Performance Comparison

| Metric | ResNet18 | ViT-B/16 |
|----------|----------|----------|
| **Accuracy** | 0.9806 | 0.9787 |
| **Precision**| 0.9806 | 0.9797 |
| **Recall**   | 0.9806 | 0.9787 |
| **F1 Score** | 0.9806 | 0.9789 |

## 2. Feature Separability Comparison

*(Note: Review the generated t-SNE plots in `outputs/tsne/` to fully analyze feature separability.)*

- **ResNet18**: Convolutional Neural Networks like ResNet18 often capture localized spatial hierarchies well. The t-SNE plot for ResNet18 should display distinct clustering based on textural and structural differences in the MRI scans.
- **ViT-B/16**: Vision Transformers capture global context through self-attention mechanisms. The t-SNE plot for ViT is expected to show potentially tighter intra-class clusters and wider inter-class margins if the global context provides a stronger discriminative signal for tumor classification.

## 3. t-SNE Clustering Analysis

- Are the four classes (Glioma, Meningioma, No Tumor, Pituitary) well-separated in the 2D projected space?
- Any overlapping clusters generally indicate morphological similarities between certain tumor types that both models struggle to distinguish. (For instance, differentiating some meningiomas from gliomas).

## 4. Conclusion: Which model learned more discriminative features?

Based on the quantitative metrics above:
- The model with the higher **F1 Score** generally maintains a better balance of precision and recall.
- Qualitatively, the model whose t-SNE visualization shows more clearly defined, separate clusters has learned stronger discriminative deep features.
