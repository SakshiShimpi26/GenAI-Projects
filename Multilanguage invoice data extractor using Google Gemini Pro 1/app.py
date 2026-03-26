from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-pro")

def get_gemini_response(input,image,prompt):
    response = model.generate_content([input,image[0],prompt])
    return response.text

def input_image_data_extractor(uploaded_file):
    if uploaded_file is not None:
        byte_data = uploaded_file.getvalue()
        image_part = [
            {
                "mime_type":uploaded_file.type,
                "data":byte_data
            }
        ]
        return image_part
    else:
        raise FileNotFoundError("File not uploaded")


st.set_page_config("Multilanguage Invoice Extractor")

st.title("Multilanguage invoice data extractor using Google Gemini Pro :)")
input = st.text_input("Input:", key='input')
uploaded_file = st.file_uploader("Choose an image of invoice....", type=['jpg','jpeg','png'])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",use_container_width=True)

submit = st.button("Tell me about the image")

input_prompt = """
You are an expert in understanding invoices. We will upload a image as invoice and you will have to answer 
any question based on the uploaded invoive accurately.
"""

if submit:
    image_data = input_image_data_extractor(uploaded_file)
    response= get_gemini_response(input_prompt,image_data,input)
    st.subheader("The extracted information is:")
    st.write(response)