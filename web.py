import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)     #Converting image to array
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)
st.sidebar.title("Potato Leaf Disease Detection")
app_mode = st.sidebar.selectbox('Select Tab', ['Home', 'Disease Recognition'])

if(app_mode == 'Home'):
    st.header("Potato Leaf Disease Detection")
    st.write("Please select 'Disease Recognition' tab to predict")
    img = Image.open('potato.jpeg')
    st.image(img)
    st.write("Plant diseases pose a significant challenge to the agricultural industry, with early identification playing a crucial role in managing infections and improving crop yield. This study focuses on developing a deep learning-based model for detecting potato leaf diseases. Various data augmentation techniques, such as flips and rotations, are applied to prevent overfitting and improve model generalization.")
    st.write("For classification, a convolutional neural network (CNN) is implemented with the Adam optimizer, known for its adaptive learning rate and efficient convergence. The model is trained on a dataset of labeled potato leaf images, allowing it to extract key features and make accurate predictions. To further enhance performance, hyperparameter tuning and regularization techniques are utilized, ensuring optimal accuracy while reducing computational overhead.")
    st.write("The model achieves a high accuracy of 96%, demonstrating its effectiveness in identifying potato leaf diseases. To validate the performance, visualization techniques are generated. These visualizations provide insights into the model’s classification capabilities, highlighting precision, recall, and overall effectiveness.")
    st.write("This study presents a robust and efficient deep learning approach for automated plant disease detection, which can assist farmers in early disease identification and timely intervention. The integration of deep learning and image processing in agriculture opens new possibilities for scalable and accurate plant disease monitoring systems, ultimately contributing to improved agricultural productivity and sustainability.")
    st.write("In this project, we classify potato leaf as: ")
    st.write("->  Early Blight")
    st.write("->  Healthy")
    st.write("->  Late Blight")
    st.write("The Visualisation of Accuracy Results can be plotted. A graph is shown to show the variation in training accuracy and validation accuaracy")
    img2 = Image.open('output.png')
    st.image(img2)

elif(app_mode == 'Disease Recognition'):
    st.header("Potato Leaf Detection System for Sustainable Agriculture")
    img = Image.open('zoom.png')
    st.image(img)
    st.write("Please select 'Browse files' to upload the desired leaf. You can view the image by clicking 'Show Image' and predict by clicking 'Predict' button. ")
    test_image=st.file_uploader("Choose an image")
    if(st.button('Show Image')):
        st.image(test_image, width=4, use_container_width=True)
    if(st.button('Predict')):
        st.snow()
        st.write('Predicting🔎🔎🔎') 
        result_index = model_prediction(test_image)
        class_name=['Potato Early Blight', 'Potato Late Blight', 'Potato Healthy']
        st.success('Model is Predicting its {}'.format(class_name[result_index]))