# Deep Learning for Brain Tumour MRI Classification

Performance, explainability and robustness to missing image information, using a fine-tuned ResNet18 on a public brain MRI dataset.

> Educational research project. Not a medical device and not validated for clinical use.

## Research question
How accurately can a deep learning model classify brain tumour MRI scans, where does it look when it decides, can its confidence be trusted, and how does accuracy change when parts of the image are hidden?

## Data
Brain Tumor MRI Dataset (Kaggle, Masoud Nickparvar): 4 classes (glioma, meningioma, pituitary, no tumour).
- Training images: [ ] (split 80% training / 20% validation)
- Test images: 1,600 (400 per class)

## Method
1. Resize images to 224 x 224 and normalise them.
2. Fine-tune a ResNet18 pretrained on ImageNet (Adam, learning rate 1e-4, batch size 32, 5 epochs, no augmentation).
3. Evaluate once on the held-out test set: accuracy, per-class precision/recall/F1, confusion matrix, ROC curves.
4. Explain predictions with Grad-CAM heatmaps.
5. Test robustness by hiding random 16 x 16 pixel patches (0% to 70% of the image) and measuring accuracy.
6. Check calibration with a reliability diagram and Expected Calibration Error (ECE).

## Key results
- **Test accuracy: 0.933** (macro F1 0.932)
- **Calibration (ECE): 0.046** (average confidence 0.979 vs accuracy 0.933, so slightly overconfident)
- **Most common mistake:** true glioma predicted as no tumour (45 images)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | 0.988 | 0.790 | 0.878 |
| Meningioma | 0.891 | 0.965 | 0.927 |
| No tumour | 0.881 | 0.998 | 0.936 |
| Pituitary | 0.995 | 0.980 | 0.987 |

![Training curves](figures/p2_training_curves.png)
![Confusion matrix](figures/p2_confusion.png)
![ROC curves](figures/p2_roc.png)

## Explainability (Grad-CAM)
Red and yellow regions are the areas that most influenced the prediction.

![Grad-CAM](figures/p2_gradcam.png)

[Write 2 to 3 sentences describing only what you actually see: do the heatmaps fall on the tumour, or on the skull edge or background?]

![Misclassified images](figures/p2_mistakes.png)

## Robustness to missing image information

| Image area hidden | Accuracy | Average confidence |
|---|---|---|
| 0% | 0.933 | 0.979 |
| 10% | 0.426 | 0.862 |
| 20% | 0.274 | 0.928 |
| 30% | 0.251 | 0.969 |
| 40% | 0.250 | 0.974 |
| 50% | 0.250 | 0.974 |
| 60% | 0.250 | 0.975 |
| 70% | 0.250 | 0.981 |

Accuracy dropped sharply once small patches were hidden and fell to 0.250 (random guessing for 4 equal classes) from 40% hidden, while the model's confidence stayed high. The model was trained without hidden regions and the grey patches differ from the black scan background, so this is a stress test and not a measure of performance on real missing MRI sequences.

![Robustness](figures/p2_robustness.png)
![Robustness examples](figures/p2_robustness_examples.png)

## Calibration
![Calibration](figures/p2_calibration.png)

[Describe whether wrong predictions tend to have lower confidence than correct ones.]

## Feature space
![t-SNE](figures/p2_tsne.png)

## Limitations
- One model, one training run, one split; variability was not measured.
- Only 5 epochs, no hyperparameter tuning, no data augmentation.
- 2D slices with no patient IDs, so possible overlap of patients between training and test sets, and duplicate images, were not checked. Scores may be optimistic.
- Public data from several sources; no external validation.
- Hidden patches are a simulation and are not the same as a missing MRI sequence.
- Grad-CAM shows where the model looked, not why it decided. It is not clinical evidence.
- Not clinically validated.

## Future work
- Train with occlusion augmentation (e.g. random erasing) and repeat the robustness test.
- Repeat with several random seeds and report variability.
- Compare more architectures and add calibration methods such as temperature scaling.
- Use patient-level splits and external validation.
- Combine MRI with clinical information and handle missing scans or missing clinical data.

## Link to Project 1
[missing-data-imputation-ml](https://github.com/YOUR-USERNAME/missing-data-imputation-ml) studies missing values in tabular clinical data. This project looks at the image equivalent: what happens when parts of a scan are missing.

## Run it
Open `brain_tumour_mri_classification.ipynb` in Google Colab, set Runtime to a T4 GPU, and run the cells from top to bottom. The dataset is downloaded automatically. Trained model weights are not included in this repository.

## Tools
Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, matplotlib, seaborn, kagglehub
