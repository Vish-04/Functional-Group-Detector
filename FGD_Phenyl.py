import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PATH = 'C:\\Py Projects\\Machine Learning\\Functional Group Detector\\Alkene DS\\phenyl_DS.csv'

# Load the data from the CSV file
df = pd.read_csv(PATH, sep=',')

# Removing empty rows
df = df.dropna()

# Convert the labels to integers
df['label'] = df['label'].astype(int)

# if diagnostics:
#   print(df.head)
#   print(df.columns)

# Load the images and labels
X = []
y = []
for index, row in df.iterrows():
  # Load the image
  img = cv2.imread(row['filename'])

  # Preprocess the image
  img = cv2.resize(img, (64, 64))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = img / 255.0

  # Add the image and label to the lists
  X.append(img)
  y.append(row['label'])

# Convert the lists to NumPy arrays
X = np.array(X).reshape(-1, 64, 64, 1)
y = np.array(y)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model architecture
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu', input_shape=(64, 64, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=18, batch_size=32)

# Evaluate the model
print("Evaluating the model... ... ...")
loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Use the model to predict whether a given image contains two parallel straight lines
predictions = model.predict(X_test)

# Loop through the test images and print the predicted class for each image
for i in range(len(X_test)):
  print("Prediction for image", i, ":", predictions[i])

# Use a threshold to determine whether a prediction is considered a positive or negative class
threshold = 0.5
correct = len(X_test)
rww = 0
wwr = 0
for i in range(len(X_test)):
  if predictions[i] > threshold:
    # print("Image", i, "guessed 1")
    if y_test[i] == 0:
      correct = correct-1
      rww = rww + 1
  if predictions[i]<threshold:
    if y_test[i] == 1:
      wwr = wwr + 1
      correct = correct -1

print("Our habibi got", correct, "out of", len(X_test), "correct!")
print("said right when wrong", rww, "times, and wrong when right", wwr, "times")

model.save('alkene_model.h5')


