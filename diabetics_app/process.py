import os
from turtle import title
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os

from pathlib import Path

from .models import Test

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_DIR= os.path.join(BASE_DIR, 'model/best_weights.hdf5')
MODEL_DIR2 = os.path.join(BASE_DIR, 'model/keras_model.h5')
WEIGHT_DIR = os.path.join(BASE_DIR, 'model/retina_weights.hdf5')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 
MODEL_DIR = os.path.join(BASE_DIR, 'model/keras_model.h5')

def process_img(self):
    patient_instance = Test.objects.get(slug=self.slug)
    DIR = str(patient_instance.get_data_url())
    DIR = (os.path.join(BASE_DIR, "plugins\media\\"+str(DIR.replace('/','\\'))))

    # # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # # Load the model
    model = tensorflow.keras.models.load_model(MODEL_DIR)

    # # Create the array of the right shape to feed into the keras model
    # # The 'length' or number of images you can put into the array is
    # # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # # Replace this with the path to your image
    image = Image.open(DIR)

    # # resize the image to a 224x224 with the same strategy as in TM2:
    # # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)

    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # # turn the image into a numpy array
    image_array = np.asarray(image)

    # # display the resized image
    # image.show()

    # # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # # Load the image into the array
    data[0] = normalized_image_array

    # # run the inference
    prediction = model.predict(data)

    # # determining predicted result
    pred_new = prediction[0]

    pred = max(pred_new)
    kind = 0

    for i in range(4):
        if pred_new[i] == pred:
            kind = i
                
    index = pred_new.tolist().index(pred)

    # #plot the graph
    import matplotlib.pyplot as plt

    # # x-coordinates of left sides of bars
    left = [1, 2, 3, 4, 5]

    # # heights of bars
    height = pred_new.tolist()
    new_height = []

    for i in height:
        new_height.append(round(i, 2) * 100)

    tick_label = ['No_DIR', 'Mild', 'Moderate', 'Sever', 'Proliferative']
    
    # plotting a bar chart
    plt.bar(left, new_height, tick_label=tick_label,
            width=0.8, color=['green'])

    # # naming the x-axis
    plt.xlabel('x - axis')
    # # naming the y-axis
    plt.ylabel('y - axis')
    # # plot title
    plt.title('Diabetic Retinopathy')

    plt.savefig(f"plugins\media\diabetics_test\{self.slug}.png")
    patient_instance.generated_data = f"diabetics_test/{self.slug}.png"
    patient_instance.result = tick_label[kind]

    patient_instance.save()

    # plt.show()
    result = []
    if index == 0:
        result.append("No DR")
    elif index == 1:
        result.append("Mild")
    elif index == 2:
        result.append("Moderate")
    elif index == 3:
        result.append("Sever")
    elif index == 4:
        result.append("Proliferative")

    accuracy = round(pred, 2)
    result.append("-")
    result.append(accuracy * 100)

    return result