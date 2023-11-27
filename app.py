import streamlit as st
import pickle
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb



def image_classification_task(model1):
    
    st.title("Tumour detection")
    uploaded_file = st.file_uploader("Upload an image for tumor detection:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Add your tumor detection code here using the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
       
        if st.button("Submit"):
            if uploaded_file is None:
                st.warning("Please upload an image first.")
            else:
                # Get the path to the uploaded file
                img_path = "temp_image.jpg"
                with open(img_path, "wb") as img_file:
                    img_file.write(uploaded_file.read())

                # Read and preprocess the image
                img = Image.open(img_path)
                img = img.resize((128, 128))
                img = np.array(img)
                input_img = np.expand_dims(img, axis=0)
        
    
                # Call the make_prediction function
                result = model1.predict(input_img)
                if result:
                    st.success("Tumor Detected")
                else:
                    st.success("No Tumor")

def spam_detection(model2):
   

    user_input = st.text_area("Enter your SMS text here:")

    if st.button("Predict"):
        # Tokenize and pad the input text
        tokeniser = Tokenizer()
        tokeniser.fit_on_texts([user_input])
        encoded_input = tokeniser.texts_to_sequences([user_input])
        padded_input = tf.keras.preprocessing.sequence.pad_sequences(encoded_input, maxlen=10, padding='post')

        # Make prediction
        prediction = model2.predict(padded_input)

        # Display result
        if prediction[0][0] > 0.5:
            st.warning("This SMS is predicted as SPAM.")
        else:
            st.success("This SMS is predicted as HAM.")
def spam_detect(model3):
        
    user_input = st.text_area("Enter your sms text here:")

    if st.button("Predict"):
        # Tokenize and pad the input text
        tokeniser = Tokenizer()
        tokeniser.fit_on_texts([user_input])
        encoded_input = tokeniser.texts_to_sequences([user_input])
        padded_input = tf.keras.preprocessing.sequence.pad_sequences(encoded_input, maxlen=10, padding='post')

        # Make prediction
        prediction = model3.predict(padded_input)

        # Display result
        if prediction[0][0] > 0.5:
            st.warning("This sms is spam.")
        else:
            st.success("This sms is ham.")
        
    
def imdb_detect(model4):
        
    user_input = st.text_area("Enter your review  here:")

    if st.button("Predict"):
        # Tokenize and pad the input text
        tokeniser = Tokenizer()
        tokeniser.fit_on_texts([user_input])
        encoded_input = tokeniser.texts_to_sequences([user_input])
        padded_input = tf.keras.preprocessing.sequence.pad_sequences(encoded_input, maxlen=10, padding='post')

        # Make prediction
        prediction = model4.predict(padded_input)

        # Display result
        if prediction[0][0] > 0.5:
            st.warning("This review is predicted as positive.")
        else:
            st.success("This review is predicted as negative.")
        
    user_input = st.text_area("Enter your review text here:")
def review(new_review_text, model5):
    max_review_length = 500
    word_to_index = imdb.get_word_index()
    new_review_tokens = [word_to_index.get(word, 0) for word in new_review_text.split()]
    new_review_tokens = pad_sequences([new_review_tokens], maxlen=max_review_length)
    prediction = model5.predict(new_review_tokens)
    print(prediction)
    prediction = "Negative" if prediction > 0.5 else "Possitive"
    st.write(f"Prediction: {prediction} ")


def main():
    st.title("Deep Learning App")

    # Dropdown for task selection
    task = st.selectbox("Select a task:", ["Tumour detection", "Sentimental classification"])

    if task == "Tumour detection":
        st.title("Tumor detection")
        model1=pickle.load(open("D:\S3\stream_model\CNN\saved_cnn.pkl","rb"))

        image_classification_task(model1)
    elif task=="Sentimental classification":
        st.title("Spam detection")
        selected_model=st.radio("Select a model",("RNN","DNN","Perceptron","Back Propogation","LSTM"))
        if selected_model=="RNN":
             model2=pickle.load(open(r"D:\S3\stream_model\CNN\saved_rnn.pkl","rb"))
             spam_detection(model2)
        if selected_model=="DNN":
             model3=pickle.load(open(r"D:\S3\stream_model\CNN\saved_dnn.pkl","rb"))
             spam_detect(model3)
        if selected_model=="Back Propogation":
             model4=pickle.load(open(r"D:\S3\stream_model\CNN\saved_back.pkl","rb"))
             imdb_detect(model4)
        if selected_model=="LSTM":
             model5=load_model(r"D:\S3\stream_model\CNN\LS.keras")
             user_input=st.text_area("Enter your review text here:")
             review(user_input,model5)
        if selected_model=="Perceptron":
             model5=load_model(r"D:\S3\stream_model\CNN\saved_per.pkl")
             user_input=st.text_area("Enter your review text here:")
             review(user_input,model5)
        
if __name__ == "__main__":
    main()

