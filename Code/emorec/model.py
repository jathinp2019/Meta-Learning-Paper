import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2

# Prompt the user to enter the folder path
folder_path ="emorec\images" 

# Check if the entered path exists
if not os.path.exists(folder_path):
    print("Invalid folder path!")
    exit()

# List all files and directories in the specified folder
contents = os.listdir(folder_path)

# Print the contents
print("Contents of the folder:")
for item in contents:
    print(item)

# Define paths
train_data_dir = folder_path  # Use the folder path for training data
batch_size = 32
num_epochs = 5
num_classes = len(contents)  # Number of classes is equal to the number of items in the folder

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),  # InceptionV3 input size
    batch_size=batch_size,
    class_mode='categorical')

# Load InceptionV3 base model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add custom classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=num_epochs, steps_per_epoch=train_generator.samples // batch_size)

# Evaluate the model
evaluation = model.evaluate(train_generator)

# Display accuracy
accuracy = evaluation[1]
print("Accuracy:", accuracy)
