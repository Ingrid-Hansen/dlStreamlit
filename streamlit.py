
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


model = load_model("modelSaved/savedmodel.tf")




selected_number = st.slider("Select a number", min_value=1, max_value=70)

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


# %%
predictions = model.predict(test_ds)

predicted_classes = np.argmax(predictions, axis=1)

y_true = np.concatenate([y for x, y in test_ds])

# %%
cm = confusion_matrix(true_labels, predicted_classes)

st.write(cm)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["forest","mountain","beach","desert","city"], yticklabels=["forest","mountain","beach","desert","city"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()





