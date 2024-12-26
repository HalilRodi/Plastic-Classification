# Plastic Classification with Carbon Footprint Estimation

This repository contains code for a plastic classification model that not only predicts the type of plastic in an image but also estimates a random carbon footprint value based on predefined ranges for each plastic type. The model uses machine learning and deep learning techniques, including VGG16, CNN, and ANN architectures.

## Features

Preprocesses images for training, validation, and testing.

Trains models using transfer learning (VGG16), CNN, and ANN architectures.

Classifies plastic types into six categories (PET, HDPE, PVC, LDPE, PP, PS).

Estimates a random carbon footprint value (kg CO2e) for the predicted plastic type.

## Dataset

The dataset used for this project can be found on Kaggle: [Plastic Classification Dataset](https://www.kaggle.com/datasets/hallrod/plastic-classification-dataset)

The dataset includes images of various types of plastics organized into labeled directories, making it suitable for supervised learning tasks.

## How It Works

Data Preprocessing: Images are resized, normalized, and augmented to improve model generalization.

## Model Training:

Classical machine learning models (SVM, Random Forest, Decision Tree) are trained using features extracted from VGG16.

Deep learning models (VGG16, CNN, ANN) are trained to classify the plastic types.

## Prediction:

The model predicts the class of the input image.

Based on the prediction, a random carbon footprint value is estimated using predefined ranges for each plastic type:

PET: 2.8 - 4.2 kg CO2e

HDPE: 1.8 - 2.0 kg CO2e

PVC: 1.9 - 3.0 kg CO2e

LDPE: 1.8 - 2.0 kg CO2e

PP: 1.7 - 2.0 kg CO2e

PS: 3.0 - 3.5 kg CO2e

## Contact

For questions or suggestions, feel free to open an issue or contact me at daghalilrodi1@gmail.com.

