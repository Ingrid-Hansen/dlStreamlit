
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import random
from PIL import Image

def display_image(category):
    # Set the path to your image folder
    folder_path = f"./images/training_set/{category}/"



    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))]

    # Choose a random image from the list
    random_image_file = random.choice(image_files)

# Display the random image in the Jupyter Notebook
    image_path = os.path.join(folder_path, random_image_file)
    return image_path








st.title("Image Classification with deep learning")
st.markdown("Ingrid Hansen - r0879034")
st.markdown("In this application you can see the results from a simple deep learning model.")
st.markdown("This model attempts to classify images of landscapes.")





image_captions = {
    'beach': 'Beautiful Beach',
    'mountain': 'Majestic Mountains',
    'forest': 'Lush Forest',
    'city': 'Cityscape',
    'desert': 'Scenic Desert'
}




st.title("The Images")


st.header("Here you can see some samples of the data the model is trained on.")
st.markdown("The image distribution among the categories.")
# Initialize lists to store image counts for training and test sets
image_train = []
image_test = []
folders = [
    'beach',
    'mountain',
    'forest',
    'city',
    'desert'
]

# Loop through the categories and count images in each folder for both training and test sets
for category in folders:
    # Training set
    train_folder_path = f"./images/training_set/{category}/"
    train_files = os.listdir(train_folder_path)
    train_image_files = [f for f in train_files if f.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))]
    train_image_count = len(train_image_files)
    image_train.append(train_image_count)

    # Test set
    test_folder_path = f"./images/test_set/{category}/"
    test_files = os.listdir(test_folder_path)
    test_image_files = [f for f in test_files if f.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))]
    test_image_count = len(test_image_files)
    image_test.append(test_image_count)

# Create a grid of subplots (1 row, 2 columns) to place the bar charts side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the bar chart for the training set on the left subplot
axes[0].bar(folders, image_train, color='purple')
axes[0].set_xlabel('Categories')
axes[0].set_ylabel('Number of Images')
axes[0].set_title('Training Set')

# Plot the bar chart for the test set on the right subplot
axes[1].bar(folders, image_test, color='orange')
axes[1].set_xlabel('Categories')
axes[1].set_ylabel('Number of Images')
axes[1].set_title('Test Set')

# Rotate the category names for better readability
axes[0].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout for better appearance
plt.tight_layout()

st.pyplot(plt)


st.header("Here you can see some samples of the data the model is trained on.")
st.markdown("Click the refresh button on the bottom to see different images.")
col1, col2 = st.columns(2)

def image_display():
    with col1:
        st.image(display_image('beach'), caption=image_captions['beach'], use_column_width=True)
        st.image(display_image('mountain'), caption=image_captions['mountain'], use_column_width=True)
        st.image(display_image('desert'), caption=image_captions['desert'], use_column_width=True)

    with col2:
        st.image(display_image('forest'), caption=image_captions['forest'], use_column_width=True)
        st.image(display_image('city'), caption=image_captions['city'], use_column_width=True)

st.markdown("Due to a bug in streamlit the images will duplicate in number after clicking refresh.")
image_display()
if st.button("Refresh Images"):
    col1.empty()  # Clear the first column
    col2.empty()  # Clear the second column
    image_display()  # Reload the images










st.title("The Confusion Matrix")
# Set the parameters for your data
batch_size = 32
image_size = (64, 64)
validation_split = 0.2

# Create the training dataset from the 'train' directory
train_ds = image_dataset_from_directory(
    directory='images/training_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)

# Create the validation dataset from the 'train' directory
validation_ds = image_dataset_from_directory(
    directory='images/training_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='validation',
    seed=123
)

# Create the testing dataset from the 'test' directory
test_ds = image_dataset_from_directory(
    directory='images/test_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size
)


model = load_model("modelSaved/layermodel.tf")



st.markdown("Here you can select an epoch to see how the confusion matrix changes throughout the training process.")
selected_number = st.slider("Select an epoch", min_value=1, max_value=70)

# Format the selected number to have leading zeros if less than 10
epoch_nr = f"{selected_number:02d}"

# Display the formatted number


model.load_weights(f"saved_models/checkpoints/layertest/weights_{epoch_nr}.tf")




# Create empty lists to store the true labels
true_labels = []

# Iterate through the test dataset to extract true labels
for images, labels in test_ds:
    true_labels.extend(np.argmax(labels, axis=1))

# Convert true_labels to a list or array
true_labels = list(true_labels)


predictions = model.predict(test_ds)

predicted_classes = np.argmax(predictions, axis=1)

y_true = np.concatenate([y for x, y in test_ds])

cm = confusion_matrix(true_labels, predicted_classes)
fig_cm = plt.figure()
plt.title('Confusion Matrix')
st.pyplot(sns.heatmap(cm, annot=True, fmt="d", cmap="Blues").get_figure())



