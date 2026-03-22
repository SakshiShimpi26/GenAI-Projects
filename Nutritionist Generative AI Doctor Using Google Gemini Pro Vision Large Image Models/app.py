from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_prompt,image):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content([input_prompt,image[0]])
    return response.text


def input_image_setup(uploded_file):
    if uploded_file is not None:
        bytes_data = uploded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")


# Streamlit UI Part
st.set_page_config("Gemini Health App")
st.header("Generative AI Doctor :)")
uploaded_file = st.file_uploader("Pick a food image ....", type=['jpg','jpeg','png'])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

submit= st.button("Tell me whether the food is healthy ?")


input_prompt = """
        You are an expert nutritionist where you need to see the food items from the image and calculate the total 
        calories, also provide the details of every food item mentioned in the image with calories intake in the 
        following format:
        1. Item 1 -- No of calories
        2. Item 2 -- No of calories
        ----
        ----
        Also mention whether the food is healthy or not.
        Also mention the percentage split ratio of carbohydrates, fats, protiens, sugar, fibers and other important 
        things required in our diet.
"""

# The actual execution happens here 
if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt,image_data)
    st.header("The Gen AI Doctor says:")
    st.write(response)

    

