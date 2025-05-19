import os
import shutil

original_dataset_path = '/content/plant_disease_dataset' #orginal_data

filtered_dataset_path = '/content/filtered_by_plant' #filtered_one
os.makedirs(filtered_dataset_path, exist_ok=True)

selected_plants = ['Tomato', 'Potato','Bell Pepper','Apple'] #selecting as per requirement

# List all class folders in the original dataset
all_class_dirs = os.listdir(original_dataset_path)

# Loop through all class folders
for class_dir in all_class_dirs:
    # Check if class_dir starts with any of the selected plant types
    if any(class_dir.startswith(plant) for plant in selected_plants):
        src = os.path.join(original_dataset_path, class_dir)
        dst = os.path.join(filtered_dataset_path, class_dir)
        # Set dirs_exist_ok=True to prevent FileExistsError if the destination directory already exists
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"✅ Copied: {class_dir}")
        #Going through all folders in the original dataset.
        #Checking if the folder name starts with one of your selected plant names.
        #If it does, you copy that folder into a new filtered folder.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = '/content/filtered_by_plant'

img_height, img_width = 128, 128 #all images are resized
batch_size = 16 #32 images are fed per training step

#loading data without augumentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  #20% for validation
)

train_generator = datagen.flow_from_directory(
    dataset_path,#[0,255] to [0,1] pixel values
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',#one-hot encoded [01000]
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, LSTM

#without overfitting control

input_shape = (img_height, img_width, 3)
num_classes = train_generator.num_classes

inputs = Input(shape=input_shape)

# CNN Layers
x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

# Reshape for RNN
x = Reshape((16*16, 128))(x)  # e.g., 16 time steps, 128 features per step

# RNN Layer
x = LSTM(64)(x)

# Output
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#FOR EARLY STOPPING

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    epochs=7,  # No. of training cycles You can adjust
    validation_data=val_generator,
     callbacks=[early_stop],
    verbose=1
)
print("✅ Training complete.")

#without addressing overfitting,data imabalance,
#without data augmentation
#by addressing these would still give more accuracy.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Get predictions and true labels
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_generator.classes

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = list(val_generator.class_indices.keys())

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Optional: print classification report
print(classification_report(y_true, y_pred, target_names=class_names))
