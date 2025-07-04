# app.py

import streamlit as st
import google.generativeai as genai
import os # Still needed for os.environ.get if running outside Streamlit Cloud directly

# --- API Key Configuration ---
# Use st.secrets for Streamlit Community Cloud deployment
# For local testing, ensure GOOGLE_API_KEY is set in your environment
# or via a .env file if you're using python-dotenv (not in this code for simplicity)
API_KEY_GEMINI = st.secrets.get("GOOGLE_API_KEY")

if not API_KEY_GEMINI:
    st.error("ERROR: Gemini API Key (GOOGLE_API_KEY) not found in Streamlit secrets.")
    st.info("Please set it in your Streamlit Cloud app's secrets or as an environment variable for local testing.")
    st.stop() # Stop the app if API key is missing

@st.cache_resource # Cache the model initialization to run only once
def initialize_gemini_model():
    try:
        genai.configure(api_key=API_KEY_GEMINI)
        # Using gemini-2.0-flash as specified in your previous backend
        model = genai.GenerativeModel('gemini-2.0-flash')
        st.success("✅ Gemini model initialized successfully.")
        return model
    except Exception as e:
        st.error(f"❌ ERROR: Failed to initialize Gemini model: {e}")
        st.info("Please check your internet connection or Gemini API key status.")
        return None

# Initialize the model globally (will be cached)
gemini_model = initialize_gemini_model()

if not gemini_model:
    st.stop() # Stop if model initialization failed

# --- Session State Management ---
# Initialize session state variables if they don't exist
if 'generated_story' not in st.session_state:
    st.session_state.generated_story = ""
if 'story_prompt_history' not in st.session_state:
    st.session_state.story_prompt_history = "" # To store the original prompt for next chapter/refine

# --- Core Generation Function ---
def generate_text_with_gemini(prompt_text, max_tokens=1000, temperature=0.8):
    """Helper function to call Gemini API."""
    if not gemini_model:
        st.error("Gemini model is not loaded. Cannot generate text.")
        return "Error: Model not available."

    try:
        response = gemini_model.generate_content(
            prompt_text,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40
            }
        )
        if hasattr(response, 'text'):
            return response.text.strip()
        else:
            safety_feedback = "No text generated."
            if response.prompt_feedback and response.prompt_feedback.safety_ratings:
                safety_feedback = f"Safety ratings: {response.prompt_feedback.safety_ratings}"
            st.warning(f"Gemini generation blocked or failed. Reason: {safety_feedback}. Try a different prompt.")
            return f"Content blocked or no text generated: {safety_feedback}"
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}. Please check your prompt or API key/quota.")
        return f"Error: {str(e)}"

# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="AI Story & Script Generator")

st.title("📖 AI Story & Script Generator")
st.markdown("""
    Generate new stories, continue chapters, refine narratives, or create scripts using **Google Gemini 2.0 Flash**.
    This app runs efficiently on CPU-only environments.
    **Remember to set your `GOOGLE_API_KEY` as a Streamlit secret!**
""")

# Input section
st.subheader("1. Generate or Continue Story")
main_prompt = st.text_input(
    "Enter your main idea or prompt for the story:",
    placeholder="e.g., 'A detective solving a mysterious case in a futuristic city'"
)

col1, col2, col3 = st.columns(3)
with col1:
    genre = st.selectbox("Genre:", ["General", "Fantasy", "Sci-Fi", "Mystery", "Horror", "Romance"])
with col2:
    length = st.selectbox("Length:", ["Short", "Medium", "Long"])
with col3:
    tone = st.selectbox("Tone:", ["Neutral", "Optimistic", "Pessimistic", "Humorous", "Serious"])

# --- Action Buttons ---
st.markdown("---")
st.subheader("Choose an Action:")
button_col1, button_col2, button_col3, button_col4 = st.columns(4)

with button_col1:
    if st.button("✨ Generate New Story ✨", use_container_width=True):
        if main_prompt:
            st.session_state.generated_story = "" # Clear previous story
            st.session_state.story_prompt_history = main_prompt # Store original prompt
            with st.spinner("Generating new story..."):
                ai_prompt = f"Generate a creative story. Genre: {genre}. Tone: {tone}. Length: {length}. Main idea: \"{main_prompt}\". Make it engaging and well-structured, formatted with paragraphs."
                st.session_state.generated_story = generate_text_with_gemini(ai_prompt)
        else:
            st.warning("Please enter a main idea/prompt to generate a new story.")

with button_col2:
    if st.button("➡️ Generate Next Chapter ➡️", use_container_width=True, disabled=not st.session_state.generated_story):
        if st.session_state.generated_story and main_prompt:
            with st.spinner("Generating next chapter..."):
                ai_prompt = f"""Continue the following story based on the context provided. Focus on developing the plot, characters, or introducing a new conflict. The new part should be {length} length.
                Previous Story Context:
                "{st.session_state.generated_story}"

                New Chapter Idea/Direction (based on original prompt): "{st.session_state.story_prompt_history}".
                """
                new_chapter = generate_text_with_gemini(ai_prompt)
                st.session_state.generated_story += "\n\n---\n\n" + new_chapter # Append new chapter
        else:
            st.warning("Generate an initial story first before continuing.")

with button_col3:
    if st.button("✏️ Refine Story ✏️", use_container_width=True, disabled=not st.session_state.generated_story):
        refinement_instructions = st.text_input("How would you like to refine the current story?", placeholder="e.g., 'Make the ending more dramatic', 'Add a new character'")
        if refinement_instructions:
            with st.spinner("Refining story..."):
                ai_prompt = f"""Given the following story, apply the user's refinement instructions. Do not generate a completely new story, but modify or enhance the provided one according to the instructions.
                Original Story to Refine:
                "{st.session_state.generated_story}"
                Refinement Instructions: "{refinement_instructions}"
                """
                st.session_state.generated_story = generate_text_with_gemini(ai_prompt)
        else:
            st.warning("Please enter refinement instructions.")

with button_col4:
    if st.button("🎬 Generate Script 🎬", use_container_width=True, disabled=not st.session_state.generated_story):
        if st.session_state.generated_story:
            with st.spinner("Generating script..."):
                script_prompt = f"""Based on the following story, generate a short dialogue script focusing on key interactions or a single scene. Format it clearly with character names followed by their lines.
                Story:
                "{st.session_state.generated_story}"
                """
                generated_script = generate_text_with_gemini(script_prompt, max_tokens=500) # Scripts are usually shorter
                st.session_state.generated_script = generated_script # Store script separately
        else:
            st.warning("Generate a story first before generating a script.")

# --- Output Section ---
st.markdown("---")
st.subheader("2. Generated Output")

# Display the main story
st.text_area(
    "Generated Story:",
    value=st.session_state.generated_story,
    height=400,
    key="story_output_area" # Unique key for text_area
)

# Display the generated script if available
if 'generated_script' in st.session_state and st.session_state.generated_script:
    st.text_area(
        "Generated Script:",
        value=st.session_state.generated_script,
        height=200,
        key="script_output_area" # Unique key for text_area
    )
