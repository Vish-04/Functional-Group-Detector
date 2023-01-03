import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

PATH = "C:\\Py Projects\\Machine Learning\\test_images\\7.png"
LOOP_PATH = "C:\\Py Projects\\Machine Learning\\test_images\\"
LABEL = 1

# Load the saved model
model = tf.keras.models.load_model('alkene_model.h5')

#Add a white background to transparent compounds
# original_image = Image.open(PATH)
# new_image = Image.new('RGB', original_image.size, (255,255,255))
# new_image.paste(original_image, mask=original_image.split())


# new_image.save(PATH)


X = []
y = []

for i in range(0, 10):

    # Load the image
    img = cv2.imread(LOOP_PATH + str(i) + ".png")

    # Preprocess the image
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0

    # Add the image and label to the lists
    X.append(img)
    y.append(1)

# Convert the lists to NumPy arrays
X = np.array(X).reshape(-1, 64, 64, 1)
y = np.array(y)

predictions = model.predict(X)

for i in range(len(X)):
  print("Prediction for image", i, ":", predictions[i])
  threshold = 0.575

  if predictions[i] > 0.5:
    print(1)
  else:
    print(0)

