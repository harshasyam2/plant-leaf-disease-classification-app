# Importing Libraries
from tensorflow import keras
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import *

model=load_model("Cifar10.h5")
output_classes=['Blight','Common Rust','Gray Leaf Spot','Healthy']

def prediction(path, model):
    img=load_img(path)
    
    img = img.resize((224,224))
    
    img=img_to_array(img)
    
    img_data = img.reshape(1, 224, 224, 3)
    
    img_data = img_data/255
   
    pred=model.predict(img_data)
    
    final_pred=np.argmax(pred)
    
    return final_pred


def calc_maize_leaf(image_path):
    pred_val=prediction(image_path, model)
    final_result=output_classes[pred_val]
    return final_result
    