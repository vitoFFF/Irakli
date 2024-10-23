import streamlit as st
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
from langchain_community.llms import Ollama
import warnings
import os
from translate import Translator

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize session state variables if they don't exist
if 'recording_llm' not in st.session_state:
    st.session_state.recording_llm = False
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'send_button_pressed' not in st.session_state:
    st.session_state.send_button_pressed = False

# Initialize Whisper model
model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")
# Define the grammar correction model
grammar_model = Ollama(model="llama3.2:1b")

# Grammar corrector
def correct_grammar(email_content):
    prompt = f"Correct the grammar of the following email, and write only the corrected email: '{email_content}'"
    response = grammar_model.invoke(prompt)
    return response

# sentiment analyzer
def sentiment(input):
    prompt2 = f"Please analyze the sentiment of the following job conversation text and categorize it as positive, negative, or neutral: '{input}'"
    response2 = grammar_model.invoke(prompt2)
    return response2

# Streamlit app setup
st.set_page_config(page_title="Email Grammar Correction Tool", layout="wide", initial_sidebar_state="collapsed")
# Centering the title and moving it up
st.markdown("<h1 style='text-align: center; margin-top: -20px;'>..oOo..</h1>", unsafe_allow_html=True)

st.sidebar.title("Menu")

# Theme selection
#theme = st.sidebar.radio("Select Theme:", ("Light", "Dark"))

# Language selection for translation
language_options = {
    "Georgian ": "ka",
    "Spanish ": "es",
    "Russian 💀": "ru",
}
selected_language = st.sidebar.radio("Choose an translation option:", list(language_options.keys()))

st.sidebar.markdown("---")

# Sidebar for audio recording parameters
#st.sidebar.subheader("Audio Parameters")
CHUNK = st.sidebar.slider("Chunk Size", min_value=512, max_value=4096, value=1024, step=512)
FORMAT = pyaudio.paInt16  # Streamlit does not support changing FORMAT dynamically
CHANNELS = 1
RATE = st.sidebar.selectbox("Sample Rate", [8000, 16000, 44100], index=1)

#st.sidebar.markdown("---")

# Sidebar for Whisper model parameters
beam_size = st.sidebar.slider("Beam Size", min_value=0, max_value=5, value=1)
process_time = st.sidebar.slider("Process Time", min_value=1, max_value=10, value=5)

# Function to translate text to the selected language
def translate_to_selected_language(text, lang_code):
    translator = Translator(to_lang=lang_code)
    translation = translator.translate(text)
    return translation

# Add a container to hold the buttons
col1, col2 = st.columns([1, 20])

with col1:
    if st.button("🎙️", use_container_width=True):
        st.session_state.recording_llm = not st.session_state.recording_llm
        if not st.session_state.recording_llm:
            # When stopping, correct grammar
            if st.session_state.transcribed_text:
                llm_response = correct_grammar(st.session_state.transcribed_text)
                st.text_area("LLM Response", llm_response, height=125)  # Display response
                
                # Translate the response to the selected language
                translated_response = translate_to_selected_language(llm_response, language_options[selected_language])
                user_input = st.text_area("Translated Response", translated_response, height=150)  # Display translated response
            else:
                st.warning("No transcribed text available.")

with col2:
    if st.button("🧠"):
        # Update the session state to indicate the send button was pressed
        st.session_state.send_button_pressed = True

# Display the transcribed text
if st.session_state.send_button_pressed and st.session_state.transcribed_text:
    sentiment_result = sentiment(st.session_state.transcribed_text)
    st.text_area("Sentiment Analysis", sentiment_result, height=525)  # Display last output

# Audio recording logic
if st.session_state.recording_llm:
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    st.write("Recording...")
    frames = []

    try:
        while st.session_state.recording_llm:
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, dtype=np.int16))

            if len(frames) >= int(RATE / CHUNK * process_time):  # Process based on slider value
                audio_data = np.concatenate(frames).astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                frames = []  # Reset frames for the next chunk

                # Transcribe audio data
                segments, info = model.transcribe(audio_data, beam_size=beam_size)

                # Store the transcribed text
                st.session_state.transcribed_text = " ".join(segment.text for segment in segments)
                st.write(st.session_state.transcribed_text)

    except (pyaudio.PyAudioError, Exception) as e:
        st.error(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        st.session_state.recording_llm = False

st.sidebar.markdown("---")
st.sidebar.write("Made with 👽🛸")
st.markdown("---")
