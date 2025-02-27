import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0
import numpy as np
import os
train_dir = '/content/drive/MyDrive/FruitQuality/dataset/train'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

model_save_path = '/content/drive/MyDrive/FruitQuality/deep_model.h5'
model.save(model_save_path)

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('/content/drive/MyDrive/FruitQuality/deep_model.h5')

# Define class labels and shelf life mapping
class_labels = list(train_generator.class_indices.keys())
shelf_life = {
    'fresh_apple_ripe': 7, 'fresh_apple_underripe': 10, 'fresh_apple_overripe': 3,
    'fresh_banana_ripe': 5, 'fresh_banana_underripe': 7, 'fresh_banana_overripe': 2,
    'fresh_orange_ripe': 10, 'fresh_orange_underripe': 14, 'fresh_orange_overripe': 5,
    'fresh_capsicum_ripe': 7, 'fresh_capsicum_underripe': 10, 'fresh_capsicum_overripe': 3,
    'fresh_bitterground_ripe': 6, 'fresh_bitterground_underripe': 8, 'fresh_bitterground_overripe': 2,
    'fresh_tomato_ripe': 8, 'fresh_tomato_underripe': 12, 'fresh_tomato_overripe': 4,
    'rotten_apple': 0, 'rotten_banana': 0, 'rotten_orange': 0,
    'rotten_capsicum': 0, 'rotten_bitterground': 0, 'rotten_tomato': 0
}

# Function to predict and format the output
def predict_fruit_quality(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    # Determine freshness
    freshness = "Fresh" if predicted_class.startswith("fresh") else "Rotten"

    # Determine ripeness (if fresh)
    ripeness = None
    if freshness == "Fresh":
        if "overripe" in predicted_class:
            ripeness = "overripe"
        elif "underripe" in predicted_class:
            ripeness = "underripe"
        elif "ripe" in predicted_class:
            ripeness = "ripe"

    # Get shelf life
    shelf_life_days = shelf_life.get(predicted_class, "Unknown")

    # Format the output
    output = {
        "Predicted Class": predicted_class,
        "Freshness": freshness,
        "Ripeness": ripeness,
        "Shelf Life (days)": shelf_life_days
    }

    return output# Test the function with a sample image
image_path = '/content/drive/MyDrive/FruitQuality/fresh_banana.jpg'
result = predict_fruit_quality(image_path)
print(result)