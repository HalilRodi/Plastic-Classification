import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image



data_dir = '/content/drive/My Drive/Plastic Classification(1)' #your dataset path
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')

data_gen_train = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
data_gen_val_test = ImageDataGenerator(rescale=1./255)

train_data = data_gen_train.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=True)

val_data = data_gen_val_test.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=False)

test_data = data_gen_val_test.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=False)

batch_size = 32


n_classes = len(train_data.class_indices.keys()) 
train_steps_per_epoch = (210 * n_classes) // batch_size
val_steps = (60 * n_classes) // batch_size
test_steps = (30 * n_classes) // batch_size

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False

vgg_feature_extractor = Sequential([base_model, Flatten()])

def extract_features(generator, sample_count):
    features = []
    labels = []
    steps = sample_count // generator.batch_size

    for step in range(steps):
        x_batch, y_batch = next(generator)
        features_batch = vgg_feature_extractor.predict(x_batch)
        features.append(features_batch)
        labels.append(np.argmax(y_batch, axis=1))

    if sample_count % generator.batch_size != 0:
        x_batch, y_batch = next(generator)
        features_batch = vgg_feature_extractor.predict(x_batch)
        features.append(features_batch)
        labels.append(np.argmax(y_batch, axis=1))

    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels



train_features, train_labels = extract_features(train_data, train_data.samples)
val_features, val_labels = extract_features(val_data, val_data.samples)
test_features, test_labels = extract_features(test_data, test_data.samples)

svm_model = SVC(kernel='linear', random_state=42, probability=True)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

svm_model.fit(train_features, train_labels)
rf_model.fit(train_features, train_labels)
dt_model.fit(train_features, train_labels)

if len(original_test_labels.shape) > 1:  
    original_test_labels = np.argmax(original_test_labels, axis=1)  


svm_predictions = svm_model.predict(test_features)
rf_predictions = rf_model.predict(test_features)
dt_predictions = dt_model.predict(test_features)


print(f"SVM Test Accuracy: {svm_model.score(test_features, original_test_labels):.4f}")
print(f"Random Forest Test Accuracy: {rf_model.score(test_features, original_test_labels):.4f}")
print(f"Decision Tree Test Accuracy: {dt_model.score(test_features, original_test_labels):.4f}")

print("\nSVM Sınıflandırma Raporu:")
print(classification_report(original_test_labels, svm_predictions))

print("\nRandom Forest Sınıflandırma Raporu:")
print(classification_report(original_test_labels, rf_predictions))

print("\nDecision Tree Sınıflandırma Raporu:")
print(classification_report(original_test_labels, dt_predictions))


vgg_model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

print("\n=== VGG Model Training Starts ===")
vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vgg_history = vgg_model.fit(
    train_data,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_data,
    validation_steps=val_steps,
    epochs=10
)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

print("\n=== CNN Model Training Starts ===")
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_history = cnn_model.fit(
    train_data,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_data,
    validation_steps=val_steps,
    epochs=10
)

train_labels = to_categorical(train_labels, num_classes=n_classes)
val_labels = to_categorical(val_labels, num_classes=n_classes)
test_labels = to_categorical(test_labels, num_classes=n_classes)

ann_model = Sequential([
    Input(shape=(train_features.shape[1],)),  
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')  
])

print("\n=== ANN Model Training Starts ===")
ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

ann_history = ann_model.fit(
    train_features, train_labels,
    batch_size=batch_size,
    validation_data=(val_features, val_labels),
    epochs=10
)



test_labels_single = np.argmax(test_labels, axis=1)  

vgg_predictions = vgg_model.predict(test_data, steps=test_steps)
cnn_predictions = cnn_model.predict(test_data, steps=test_steps)
ann_predictions = ann_model.predict(test_features, batch_size=batch_size)

vgg_predicted_classes = np.argmax(vgg_predictions, axis=1)
cnn_predicted_classes = np.argmax(cnn_predictions, axis=1)
ann_predicted_classes = np.argmax(ann_predictions, axis=1)

print("\nVGG Model Sınıflandırma Raporu:")
print(classification_report(test_labels_single, vgg_predicted_classes))

print("\nCNN Model Sınıflandırma Raporu:")
print(classification_report(test_labels_single, cnn_predicted_classes))

print("\nANN Model Sınıflandırma Raporu:")
print(classification_report(test_labels_single, ann_predicted_classes))

def plot_confusion_matrix(model, data, model_name, test_labels_single=None, class_labels=None):
    predictions = model.predict(data)
    predicted_classes = predictions.argmax(axis=1)

    if test_labels_single is not None:
        true_classes = test_labels_single
    else:
        true_classes = data.classes

    if class_labels is None:
        class_labels = list(data.class_indices.keys())

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

test_labels_single = np.argmax(test_labels, axis=1) 
class_labels = list(test_data.class_indices.keys()) 

plot_confusion_matrix(vgg_model, test_data, "VGG Model", class_labels=class_labels)
plot_confusion_matrix(cnn_model, test_data, "CNN Model", class_labels=class_labels)
plot_confusion_matrix(ann_model, test_features, "ANN Model", test_labels_single=test_labels_single, class_labels=class_labels)

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_training_history(vgg_history, "VGG Model")
plot_training_history(cnn_history, "CNN Model")
plot_training_history(ann_history, "ANN Model")

image_path = input("Lütfen plastik türünü tahmin etmek için bir görselin yolunu giriniz: ")

def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  
    return np.expand_dims(img_array, axis=0)  

processed_image = preprocess_image(image_path)

carbon_footprint_ranges = {
    "PET": (2.8, 4.2),
    "HDPE": (1.8, 2.0),
    "PVC": (1.9, 3.0),
    "LDPE": (1.8, 2.0),
    "PP": (1.7, 2.0),
    "PS": (3.0, 3.5),
}

def predict_and_calculate_footprint_with_random_range(model, processed_image, model_name, class_labels):
    prediction = model.predict(processed_image)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]

    if predicted_class in carbon_footprint_ranges:
        carbon_range = carbon_footprint_ranges[predicted_class]
        estimated_footprint = np.random.uniform(*carbon_range)
    else:
        estimated_footprint = "Bilinmiyor"

    print(f"\n{model_name} Tahmini:")
    print(f"Tahmin Edilen Sınıf: {predicted_class} ({predicted_class_idx})")
    print(f"Güven: {confidence:.2f}")
    print(f"Tahmini Karbon Ayak İzi: {estimated_footprint:.2f} kg CO2" if isinstance(estimated_footprint, float) else "Bilinmiyor")
    return predicted_class, confidence, estimated_footprint

print("\n=== Modellerle Tahmin Sonuçları ===")
class_labels = list(test_data.class_indices.keys())  

predict_and_calculate_footprint_with_random_range(vgg_model, processed_image, "VGG Model", class_labels)
predict_and_calculate_footprint_with_random_range(cnn_model, processed_image, "CNN Model", class_labels)
predict_and_calculate_footprint_with_random_range(ann_model, processed_image, "ANN Model", class_labels)
