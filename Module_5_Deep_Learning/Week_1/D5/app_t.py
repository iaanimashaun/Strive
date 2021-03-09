import streamlit as st
import tensorflow as tf 
import numpy as np 
import streamlit as st 
from PIL import Image 
import requests 
from io import BytesIO 
import torch
from tf.keras.model import Sequential
from tf.keras.layers import Dense
from torchvision import transforms

p = transforms.Compose([transforms.Scale((48,48))])

st.title("Fashion Image Classification")
#st.header("Fashoin Image Classifier")
#st.text("Upload a fashion image")



st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Fashion Image Classifier')
st.text('Provide url of image for classification or Upload a fashion image')

@st.cache(suppress_st_warning=True)
def load_model(PATH):
  model =  torch.load(PATH)
  model.eval()
  return model

with st.spinner('Loading Model Into Memory.....'):
    PATH = './model'
    model = load_model(PATH)

classes= ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

def scale(image):
    p = transforms.Compose([transforms.Scale((28,28))])

    trainTransform  = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Scale((28,28)) 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open('img.jpg') / 255
    return p(img)


def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = scale(img)
  return np.expand_dims(img, axis=0)

path = None#st.text_input('Enter Image url to classify..','https://cdna.lystit.com/photos/anthropologie/e04362fd/cloth-stone-RED-Tulane-Shirt.jpeg')
if path is not None:
  content = requests.get(path).content
else:
  content = st.file_uploader("Choose a fashion ...", type="jpg")

  st.write('Predicted Class: ')
  with st.spinner('classifying......'):
    label = np.argmax(model.predict(decode_img(content)), axis=1)
    st.write('')
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Fashion Image', use_column_width=True)
    







"""
uploaded_file = st.file_uploader("Choose a fashion ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'brain_tumor_classification.h5')
    if label == 0:
        st.write("This is an ankle boot")
    else:
        st.write("This is a shirt")"""




        """"model = keras.Sequential()
model.add(Dense(32, activation='relu', input_shape=(28,28)))
model.add(Dense(10, activation='softmax'))
model.com""""




def import_and_predict(image_data, model):
    