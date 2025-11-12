

# Brain Tumor Segmentation Using Deep Learning

This project implements a **deep learning-based approach for brain tumor segmentation** in MRI images using a U-Net architecture with a pre-trained ResNet50 encoder. The model aims to accurately identify and isolate brain tumors, assisting in medical image analysis and potentially aiding radiologists in diagnosis.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Data Exploration](#data-exploration)
* [Model Architecture](#model-architecture)
* [Preprocessing & Augmentation](#preprocessing--augmentation)
* [Training](#training)
* [Evaluation & Visualization](#evaluation--visualization)
* [Usage](#usage)
* [References](#references)

---

## Project Overview

The goal of this project is to build a **semantic segmentation model** to detect brain tumors in MRI images. Semantic segmentation involves classifying each pixel of an image into predefined categories—in this case, tumor vs. non-tumor regions.

Key features:

* Uses **deep learning** for accurate segmentation.
* Employs **transfer learning** with a pre-trained ResNet50 as the encoder.
* Handles class imbalance with a custom **Dice coefficient loss function**.
* Provides **visualizations** for qualitative evaluation.

---

## Dataset

The project uses the `kaggle_3m` dataset, which contains MRI images and corresponding segmentation masks.

* **Data File:** `data_mask.csv`
* **Columns:**

  * `patient_id`: Unique identifier for each patient
  * `image_path`: Path to the MRI image
  * `mask_path`: Path to the corresponding tumor mask
  * `mask`: Binary label (0 = no tumor, 1 = tumor)
* **Number of entries:** 3,929

  * Images without tumors: 2,556
  * Images with tumors: 1,373

**Note:** The dataset is **imbalanced**, so specialized loss functions like Dice coefficient are used to mitigate this during training.

---

## Data Exploration

Initial exploration of the dataset includes:

* Checking image and mask distribution
* Visualizing sample MRI images and masks
* Identifying data imbalance and class distribution

```python
# Example: Visualizing sample MRI images and masks
import matplotlib.pyplot as plt
import cv2

image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("MRI Image")
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.show()
```

---

## Model Architecture

The model uses a **U-Net convolutional neural network**, which is widely adopted for biomedical image segmentation tasks.

**Key points:**

* **Encoder:** Pre-trained ResNet50
* **Input size:** `(256, 256, 3)`
* **Total layers:** 175
* **Total parameters:** ~23 million
* **Purpose of U-Net:** Combines encoding (feature extraction) with decoding (upsampling) to produce precise pixel-wise segmentation maps.

---

## Preprocessing & Augmentation

Images are preprocessed before training:

* **Rescaling:** Pixel values are normalized
* **Data augmentation:** Using `ImageDataGenerator` to apply transformations like rotation, flipping, and zoom to increase dataset diversity.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
```

---

## Training

**Configuration:**

* **Loss function:** `dice_coef_loss` – effective for imbalanced datasets and measures overlap between predicted and ground truth masks.
* **Optimizer:** Adam
* **Metrics:**

  * Dice Coefficient (`dice_coef`)
  * Jaccard Index (`jaccard_coef`)

**Training steps:**

1. Load images and masks
2. Split data into training and validation sets
3. Fit the model with data generators
4. Monitor metrics for performance evaluation

---

## Evaluation & Visualization

To qualitatively assess model performance, the notebook visualizes results by overlaying predicted masks on MRI images.

**Visualization steps:**

1. Original MRI image
2. Original (ground truth) mask
3. MRI with mask overlaid in **red**
4. Predicted AI mask
5. MRI with predicted mask overlaid in **green**

```python
plt.imshow(image)
plt.imshow(pred_mask, alpha=0.5, cmap='Greens')  # overlay predicted mask
plt.title("Predicted Mask Overlay")
plt.show()
```

This helps in assessing whether the model correctly identifies tumor regions.

---

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook Untitled2.ipynb
```

4. Modify paths in `data_mask.csv` if needed and start training or evaluating the model.

---

## References

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [Kaggle Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets)
* TensorFlow/Keras Documentation

---

This README provides a **comprehensive overview** of the project, making it clear for collaborators, recruiters, or anyone exploring your GitHub repository.

---


