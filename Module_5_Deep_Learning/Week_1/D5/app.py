import streamlit as st
from img_classification import teachable_machine_classification
import tensorflow as tf
import tensorflow as tf 
import numpy as np 
import streamlit as st 
from PIL import Image 
import requests 
from io import BytesIO 


st.title("Fashion Image Classification")
#st.header("Fashoin Image Classifier")
#st.text("Upload a fashion image")



st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Fashion Image Classifier')
st.text('Provide url of image for classification or Upload a fashion image')

#@st.cache(suppress_st_warning=True)
def load_model():
  #model = pickle.load(open('./model.sav', 'rb'))
  model = tf.keras.models.load_model(r'C:\Users\Amash\Google Drive\Strive\Exercises\Module_5_Deep_Learning\Week_1\D5\fashion_classifier.hdf5')
  return model

with st.spinner('Loading Model Into Memory.....'):
  model = load_model()

classes= ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

def scale(image):
  image = tf.cast(image, tf.float32)
  image /= 255.0
  return tf.image.resize(image, [28,28])

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = scale(img)
  return np.expand_dims(img, axis=0)

def preprocess(image):
  img = image.load_img(path, target_size=(28,28))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)


#path = st.text_input('Enter Image url to classify..','https://cdna.lystit.com/photos/anthropologie/e04362fd/cloth-stone-RED-Tulane-Shirt.jpeg')
if path is not None:
  #content = requests.get(path).content
#else:
  content = st.file_uploader("Choose a fashion ...", type=["jpg",'png'])

st.write('Predicted Class: ')
with st.spinner('classifying......'):
    label = np.argmax(model.predict(decode_img(content)), axis=1)
    st.write('')
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Fashion Image', use_column_width=True)








