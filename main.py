import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import splitfolders
import itertools

PATH = 'oxford-iiit-cats-extended-10k/CatBreedsRefined-v3'
data_dir = 'cats_dataset'
inputs_dir = 'inputs'
saved_model_path = 'saved_model.keras'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INPUT_SHAPE = IMAGE_SIZE + (3,)
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dir = 'data/train'
validation_dir = 'data/val'

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomFlip('vertical'),
    layers.RandomRotation(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2)
])


def get_image_paths_and_labels():
    image_paths = []
    labels = []
    for breed_dir in os.listdir(data_dir):
        breed_path = os.path.join(data_dir, breed_dir)
        if os.path.isdir(breed_path):
            for filename in os.listdir(breed_path):
                file_path = os.path.join(breed_path, filename)
                image_paths.append(file_path)
                labels.append(breed_dir)
    return image_paths, labels


def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label


def load_images_and_labels():
    image_paths, labels = get_image_paths_and_labels()
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)

    visualize_data_classes(image_paths, labels, label_encoder)

    return dataset, label_encoder


def visualize_data_classes(image_paths, labels, label_encoder):
    unique_labels = np.unique(labels, axis=0)
    fig, axes = plt.subplots(2, 6, figsize=(10, 5))
    axes = axes.flatten()
    for ax, unique_label in zip(axes, unique_labels):
        label_index = np.argmax(unique_label)
        label_name = label_encoder.inverse_transform([label_index])[0]
        image_path = next(path for path, label in zip(image_paths, labels) if np.argmax(label) == label_index)
        img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        ax.imshow(img / 255.0)
        ax.set_title(label_name)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_data_augmentation(train_dataset):
    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 8))
        first_image = image[0]
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')
    plt.tight_layout()
    plt.show()


def preprocess_labels(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)
    return labels, label_encoder


def create_model(num_classes):
    rescale = layers.Rescaling(1. / 127.5, offset=-1)

    base_model = applications.MobileNetV2(include_top=False, input_shape=INPUT_SHAPE, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        data_augmentation,
        rescale,
        base_model,
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def fine_tune_model(model):
    model.trainable = True

    fine_tune_di = 100

    for layer in model.layers[:fine_tune_di]:
        layer.trainable = False

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_evaluate(model, train_dataset, test_dataset, label_encoder, epochs=20, fine_tune_epochs=10):
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.7,
        patience=2,
        verbose=1,
        mode="max",
        min_lr=0.00001
    )
    early_stopping = callbacks.EarlyStopping(
        patience=6,
        monitor="val_accuracy",
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[lr_scheduler]
    )

    model = fine_tune_model(model)

    fine_tune_history = model.fit(
        train_dataset,
        epochs=fine_tune_epochs,
        validation_data=test_dataset,
        callbacks=[lr_scheduler, early_stopping]
    )

    test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
    print(f'Test accuracy: {test_acc}')

    plot_training_history(history, fine_tune_history)
    evaluate_model(model, test_dataset, label_encoder)


def plot_training_history(history, fine_tune_history):
    acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
    val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
    loss = history.history['loss'] + fine_tune_history.history['loss']
    val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()


def evaluate_model(model, dataset, label_encoder):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    plot_confusion_matrix(cm, label_encoder.classes_)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.colormaps['winter'])
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def predict_image(model, label_encoder):
    image_files = [os.path.join(inputs_dir, file) for file in os.listdir(inputs_dir) if file.endswith('.jpeg')]
    plt.figure(figsize=(10, 8))
    for i, image_file in enumerate(image_files):
        img = tf.io.read_file(image_file)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = img / 255.0
        img = tf.expand_dims(img, axis=0)
        predictions = model.predict(img, batch_size=10)
        predicted_index = np.argmax(predictions)
        confidence_score = predictions[0][predicted_index]

        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        plt.subplot(2, 4, i + 1)
        plt.imshow(tf.squeeze(img))
        plt.title(f"{predicted_label}\nConfidence: {confidence_score:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    dataset, label_encoder = load_images_and_labels()

    splitfolders.ratio(PATH, output='data', seed=64, ratio=(0.8, 0.2))

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        label_mode='int'
    )
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        validation_dir,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        label_mode='int'
    )
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    class_names = sorted(os.listdir(train_dir))
    class_indices = {class_name: index for index, class_name in enumerate(class_names)}

    print("Class Names:", class_names)
    print("Class Indices:", class_indices)

    if len(dataset) == 0:
        raise ValueError("No images were loaded. Check the data directory and file names.")

    labels, label_encoder = preprocess_labels(label_encoder.classes_)
    num_classes = len(label_encoder.classes_)

    print(f"The number of classes are: {num_classes}")

    visualize_data_augmentation(train_dataset)

    if os.path.exists(saved_model_path):
        model = tf.keras.models.load_model(saved_model_path)
        print("Loaded saved model.")
    else:
        model = create_model(num_classes)
        train_and_evaluate(model, train_dataset, test_dataset, label_encoder, epochs=200, fine_tune_epochs=100)

    model.save(saved_model_path)

    predict_image(model, label_encoder)


if __name__ == "__main__":
    main()
