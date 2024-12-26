# Plastic Classification with Carbon Footprint Estimation

This repository contains the code for a plastic classification model that predicts the type of plastic in an image and estimates a random carbon footprint value based on predefined ranges for each plastic type. The model utilizes machine learning and deep learning techniques, including VGG16, CNN, and ANN architectures.

## Features

- **Data Preprocessing:** Images are resized, normalized, and augmented to improve model generalization.
- **Model Training:**
  - Transfer learning using VGG16.
  - Deep learning models (CNN, ANN).
  - Classical machine learning models (SVM, Random Forest, Decision Tree) trained using features extracted from VGG16.
- **Plastic Classification:** Classifies images into six categories:
  - PET, HDPE, PVC, LDPE, PP, PS.
- **Carbon Footprint Estimation:** Provides a random carbon footprint value (in kg CO2e) based on predefined ranges for the classified plastic type.

## Dataset

The dataset used for this project is available on Kaggle: [Plastic Classification Dataset](https://www.kaggle.com/datasets/hallrod/plastic-classification-dataset).  
It contains images of various types of plastics organized into labeled directories, making it ideal for supervised learning tasks.

## How It Works

1. **Data Preprocessing:** 
   - Images are resized, normalized, and augmented to improve the model's generalization capabilities.
   
2. **Model Training:** 
   - Models are trained using both classical machine learning techniques and deep learning architectures (e.g., VGG16, CNN, ANN).
   
3. **Prediction:** 
   - The trained model predicts the plastic type based on the input image.
   - A random carbon footprint value is estimated for the predicted plastic type using the predefined ranges below.

## Carbon Footprint Ranges

| Plastic Type | Carbon Footprint Range (kg CO2e) |
|--------------|----------------------------------|
| PET          | 2.8 - 4.2                        |
| HDPE         | 1.8 - 2.0                        |
| PVC          | 1.9 - 3.0                        |
| LDPE         | 1.8 - 2.0                        |
| PP           | 1.7 - 2.0                        |
| PS           | 3.0 - 3.5                        |

## Contact

For questions, suggestions, or contributions, feel free to open an issue or reach out via email: [daghalilrodi1@gmail.com](mailto:daghalilrodi1@gmail.com).
