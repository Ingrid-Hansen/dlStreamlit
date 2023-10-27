# %% [markdown]
# 

# %%
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


NUM_CLASSES = 5

# Create a sequential model with a list of layers
model = tf.keras.Sequential([
  layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(32, (3, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation="relu"),
  layers.Dense(NUM_CLASSES, activation="softmax") # because it's one class, we use sigmoid
])



# %%
# Compile and train your model as usual
model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

print(model.summary())

# %% [markdown]
# ### Data Preprocessing and augmentation

# %% [markdown]
# 

# %%
NUM_CLASSES = 5
IMG_SIZE = 64
# There is no shearing option anymore, but there is a translation option
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

# Create a sequential model with a list of layers
model = tf.keras.Sequential([
  # Add a resizing layer to resize the images to a consistent shape
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  # Add a rescaling layer to rescale the pixel values to the [0, 1] range
  layers.Rescaling(1./255),
  # Add some data augmentation layers to apply random transformations during training
  layers.RandomFlip("horizontal"),
  layers.RandomTranslation(HEIGTH_FACTOR,WIDTH_FACTOR),
  layers.RandomZoom(0.2),


  layers.RandomFlip("vertical"),
  
  layers.RandomContrast(0.2),

  #layers.RandomCrop(32,32),
  #layers.RandomBrightness(0.2),

  



  layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(32, (3, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Flatten(), # Or, layers.GlobalAveragePooling2D()
  layers.Dense(128, activation="relu"),
  layers.Dense(NUM_CLASSES, activation="softmax")
])
checkpoint = ModelCheckpoint("saved_models/checkpoints/standardmodel/weights_{epoch:02d}.tf", save_weights_only=True, save_freq="epoch")
# Compile and train your model as usual
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# %%
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


# %% [markdown]
# ### Training and Validation

# %%
history = model.fit(train_ds,
                validation_data = validation_ds,
                steps_per_epoch = 41,
                epochs = 50,
                callbacks=[checkpoint]
                )

# %% [markdown]
# ### Validation

# %%
model.save("modelSaved/savedmodel.tf")

# %%
from keras.models import load_model

model = load_model("modelSaved/savedmodel.tf")

model.load_weights(f"saved_models/checkpoints/standardmodel/weights_48.tf")

# %%
# Create a figure and a grid of subplots with a single call
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

# Plot the loss curves on the first subplot
ax1.plot(history.history['loss'], label='training loss')
ax1.plot(history.history['val_loss'], label='validation loss')
ax1.set_title('Loss curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot the accuracy curves on the second subplot
ax2.plot(history.history['accuracy'], label='training accuracy')
ax2.plot(history.history['val_accuracy'], label='validation accuracy')
ax2.set_title('Accuracy curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Adjust the spacing between subplots
fig.tight_layout()

# Show the figure
plt.show()

# %%
# Extract training and validation accuracy from the history object
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Calculate error (misclassification rate) by subtracting accuracy from 1
training_error = [1 - acc for acc in training_accuracy]
validation_error = [1 - acc for acc in validation_accuracy]

# Create a list of epoch numbers for x-axis
epochs = range(1, len(training_accuracy) + 1)

# Plot the training and validation error
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_error, 'b', label='Training Error')
plt.plot(epochs, validation_error, 'r', label='Validation Error')
plt.title('Training and Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### Prediction

# %%
epoch_nr = 48

model.load_weights(f"saved_models/checkpoints/standardmodel/weights_{epoch_nr}.tf")


# %%
def predict_image(img):
    test_image = image.load_img(img, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0  # Ensure pixel values are in the [0, 1] range
    test_image = np.expand_dims(test_image, axis=0)
    class_names = train_ds.class_names
    # Use the model to predict the class
    predict = model.predict(test_image)

    predicted_class_index = np.argmax(predict[0])

    predicted_class_name = class_names[predicted_class_index]

    return print(f"The model predicts class: {predicted_class_name}")

to_predict = "images/PredictionImages/bondi-beach-sydney--a246321695.jpg"

predict_image(to_predict)


# %% [markdown]
# ### Confusion Matrix

# %%
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["forest","mountain","beach","desert","city"], yticklabels=["forest","mountain","beach","desert","city"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# ### Teachable Machine

# %% [markdown]
# Looking at the confusion matrix, it is clear that the model struggles the most to differenciate beach from the other categories, especially desert and forest. This is not too strange, as deserts and beaches both consist of sand, and beaches are often in close proimity to plants and cities.
# 
# 
# Since this is the area where the model struggles the most it would be interesting to see how Googles Teachable Machine performes on the beach test data. A couple of images from the test set has been selected and we will in this section see what the model created in this task predicts vs Google Teachable Machine.
# 

# %%
predict_image("images/CompareImages/beach_67.jpg")


