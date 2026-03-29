import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
import re
import google.generativeai as genai 
from youtube_transcript_api import YouTubeTranscriptApi

os.getenv("GOOGLE_API_KEY")

prompt = """ You are a youtube video summarizer. You will be taking the entire video transcript text and summarizing the 
entire video and providing important points within 250 words. Also highlight and explain the important point in point-wise manner
The transcript will be appended here: """

def generate_gemini(transcript, prompt):
    if not isinstance(transcript, str):
        return "Invalid transcript data."

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt + transcript)
    return response.text

def get_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_url):
    video_id = get_video_id(video_url)
    if not video_id:
        return None

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return "\n".join([entry["text"] for entry in transcript])
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# video_url = "https://youtu.be/HFfXvfFe9F8?si=V-n6uDBrbaNCJ92J"
# transcript = get_transcript(video_url)
# x = generate_gemini(transcript,prompt)
# print(x)

st.title("Notes from Youtube Video")
text = st.text_input("Enter the Youtube video URL")
if text:
    video_id = get_video_id(text)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

if st.button("Get Notes"):
    transcript = get_transcript(text)

    if not transcript:
        st.error("❌ Could not fetch transcript. Video may not have captions.")
    else:
        summary = generate_gemini(transcript, prompt)
        st.markdown("## Detailed Notes")
        st.markdown(
            f"""
            <div style="text-align: justify;">
                {summary}
            </div>
            """,
            unsafe_allow_html=True
        )