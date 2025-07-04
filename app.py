# --- IMPORTANT MODIFICATIONS FOR STREAMLIT NATIVE UI & SECRETS ---
# 1. Removed Gradio imports and components, replaced with Streamlit (st.) equivalents.
# 2. Replaced os.environ.get() with st.secrets.get() for API key access.
# 3. Forced Stable Diffusion to run on CPU as Streamlit Community Cloud is CPU-only.
# ------------------------------------------------------------------

import os
import torch
import streamlit as st # Only Streamlit needed for UI
from diffusers import StableDiffusionPipeline
from PIL import Image
from huggingface_hub import login
import google.generativeai as genai

# Global variables for models (initialized to None)
stable_diffusion_pipe = None
gemini_model = None

# --- Function to configure APIs and load models ---
@st.cache_resource # Use Streamlit's caching to load models only once
def initialize_models_and_apis():
    global stable_diffusion_pipe, gemini_model

    st.write("### Initializing Application (This happens once on startup)...")
    status_placeholder = st.empty() # Placeholder for status messages

    status_placeholder.info("1. Configuring API keys from Streamlit secrets...")
    # Configure Google Gemini API from Streamlit Secrets
    API_KEY_GEMINI = st.secrets.get('GOOGLE_API_KEY')
    if not API_KEY_GEMINI:
        status_placeholder.error("ERROR: Google Gemini API Key (GOOGLE_API_KEY) not found in Streamlit secrets. Text generation will not work.")
        gemini_model = None
    else:
        try:
            genai.configure(api_key=API_KEY_GEMINI)
            status_placeholder.success("✅ Gemini API configured successfully from Streamlit secrets.")
            # --- DIAGNOSTIC: List available Gemini models ---
            st.info("Checking available Gemini models...")
            target_gemini_model_name = 'models/gemini-1.5-flash-latest'
            found_target_gemini_model = False
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        if m.name == target_gemini_model_name:
                            found_target_gemini_model = True
            except Exception as e:
                st.warning(f"      ❌ Could not list models: {e}. Proceeding with caution.")
            if not found_target_gemini_model:
                st.warning(f"⚠️ WARNING: '{target_gemini_model_name}' was not found for generateContent. Text gen may fail.")
            else:
                st.success(f"✅ '{target_gemini_model_name}' found and available.")
        except Exception as e:
            status_placeholder.error(f"❌ ERROR: Gemini API configuration failed: {e}. Please verify your 'GOOGLE_API_KEY' secret.")
            API_KEY_GEMINI = None
            gemini_model = None

    # Configure Hugging Face Login from Streamlit Secrets
    HF_TOKEN = st.secrets.get('HF_TOKEN')
    if not HF_TOKEN:
        status_placeholder.error("❌ ERROR: Hugging Face Token (HF_TOKEN) not found in Streamlit secrets.")
        st.warning("Stable Diffusion model loading might fail for private models.")
    else:
        try:
            login(HF_TOKEN)
            status_placeholder.success("✅ Hugging Face login successful from Streamlit secrets ('HF_TOKEN').")
        except Exception as e:
            status_placeholder.error(f"❌ ERROR: Hugging Face login failed: {e}")
            st.warning("Stable Diffusion model loading might fail. Please verify your 'HF_TOKEN' secret.")
            HF_TOKEN = None

    status_placeholder.info("2. Loading Generative Models...")

    # Stable Diffusion (for image generation)
    # Determine device (GPU or CPU)
    sd_device = "cpu" # Force CPU as Streamlit Community Cloud is primarily CPU-based
    sd_dtype = torch.float32 # Use float32 for CPU for compatibility/stability
    st.info(f"   Stable Diffusion will use: {sd_device.upper()} (dtype: {sd_dtype}).")
    st.warning("⚠️ Running Stable Diffusion on CPU (Streamlit Community Cloud). Image generation will be VERY slow and may time out.")

    try:
        stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=sd_dtype,
            use_safetensors=True
        ).to(sd_device)
        status_placeholder.success("✅ Stable Diffusion v1.5 model loaded successfully.")
    except Exception as e:
        status_placeholder.error(f"❌ ERROR: Failed to load Stable Diffusion model: {e}")
        st.error("Image generation will not work. This is often due to insufficient RAM for CPU inference on large models or connectivity issues.")
        stable_diffusion_pipe = None

    # Gemini Flash (for text generation)
    if API_KEY_GEMINI:
        try:
            gemini_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
            status_placeholder.success("✅ Gemini Flash model ('models/gemini-1.5-flash-latest') loaded successfully.")
        except Exception as e:
            status_placeholder.error(f"❌ ERROR: Failed to load Gemini Flash model: {e}")
            st.warning("Text generation will not work. Check your internet connection or Gemini API key status.")
            gemini_model = None
    else:
        st.warning("Gemini Flash model not loaded due to missing/failed API key configuration.")
        gemini_model = None

    status_placeholder.empty() # Clear the status messages once done
    st.write("### Application Ready! Enter your prompt below.")
    return stable_diffusion_pipe, gemini_model

# Call the initialization function at the start of the app
stable_diffusion_pipe, gemini_model = initialize_models_and_apis()


# --- Function to generate image and text ---
def generate_content_streamlit(prompt_text, text_type, custom_story_text, use_custom_text):
    """
    Generates an image using Stable Diffusion and a story/monologue using Gemini.
    Provides feedback directly to Streamlit UI.
    """
    generated_image = None
    generated_story_text = ""
    status_messages = []

    # --- Image Generation (Stable Diffusion) ---
    if stable_diffusion_pipe:
        status_messages.append("🖼️ Attempting image generation (this will be slow on CPU)...")
        try:
            generated_image = stable_diffusion_pipe(prompt_text).images[0]
            status_messages.append("✅ Image generated successfully (but might have taken a long time).")
        except Exception as e:
            error_msg = f"❌ ERROR: Image generation failed: {e}. This is common on CPU due to performance/memory. Try a simpler prompt or ensure enough RAM."
            status_messages.append(error_msg)
            generated_image = None
    else:
        error_msg = "❌ Image generation skipped: Stable Diffusion model not loaded. Check initial setup and logs."
        status_messages.append(error_msg)


    # --- Text Generation (Gemini Flash) ---
    if gemini_model:
        status_messages.append("📝 Attempting text generation with Gemini...")
        text_source_prompt = custom_story_text if use_custom_text and custom_story_text.strip() else prompt_text
        if not text_source_prompt.strip():
            error_msg = "❌ Text generation skipped: No valid prompt or custom text provided for story/monologue."
            status_messages.append(error_msg)
            generated_story_text = error_msg
        else:
            if text_type == "story":
                full_gemini_prompt = f"Write a short, compelling, and family-friendly story (beginning, middle, end) based on this idea:\n\nIdea: {text_source_prompt}\n\nStory:"
            else: # monologue
                full_gemini_prompt = f"Imagine the inner thoughts of a character experiencing or observing the following scene. Write a coherent and family-friendly internal monologue:\n\nScene: {text_source_prompt}\n\nMonologue:"

            generation_config = {
                "max_output_tokens": 800,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40
            }

            try:
                response = gemini_model.generate_content(
                    full_gemini_prompt,
                    generation_config=generation_config
                )
                if hasattr(response, 'text'):
                    generated_story_text = response.text.strip()
                    status_messages.append("✅ Story/Monologue generated successfully by Gemini.")
                else:
                    safety_feedback = "No text generated."
                    if response.prompt_feedback and response.prompt_feedback.safety_ratings:
                        safety_feedback = f"Safety ratings: {response.prompt_feedback.safety_ratings}"
                    generated_story_text = f"❌ Gemini generation blocked or failed. Reason: {safety_feedback}. Please try a different prompt or adjust parameters."
                    status_messages.append(generated_story_text)

            except Exception as e:
                error_msg = f"❌ ERROR: Gemini text generation failed: {e}. Check your prompt or API key configuration."
                status_messages.append(error_msg)
                generated_story_text = error_msg
    else:
        error_msg = "❌ Text generation skipped: Gemini model not loaded or API key missing. Check GOOGLE_API_KEY secret."
        status_messages.append(error_msg)
        generated_story_text = error_msg

    return generated_image, generated_story_text, status_messages


# --- Streamlit UI Layout ---
st.set_page_config(layout="wide") # Use wide layout for better display

st.title("🖼️ Image & Story/Monologue Generator")
st.markdown("""
    Generate images with **Stable Diffusion** and accompanying stories/monologues with **Google Gemini 1.5 Flash**.
    **Ensure you've set your `GOOGLE_API_KEY` and `HF_TOKEN` as Streamlit secrets!**
""")

# Persistent status messages for generation
status_area = st.empty()

# Inputs Column
with st.container():
    st.subheader("Your Input")
    prompt_input = st.text_input(
        label="Enter Prompt for Image",
        placeholder="e.g., 'a majestic dragon flying over a medieval castle at sunset', 'a cozy reading nook with a cat sleeping'"
    )
    text_type = st.radio(
        label="Choose Text Feature",
        options=["story", "monologue"],
        index=0, # Default to 'story'
        help="Select whether to generate a story or an inner monologue."
    )
    use_custom_text = st.checkbox(
        label="Use different text for story/monologue?",
        value=False,
        help="If checked, the story/monologue will be based on 'Custom Text' instead of the image prompt."
    )
    custom_story_input = st.text_area(
        label="Custom Text for Story/Monologue (Optional)",
        placeholder="e.g., 'a detective solving a mysterious case', 'the thoughts of a robot exploring a new planet'",
        disabled=not use_custom_text # Disable if checkbox is not ticked
    )

    # Enable/disable custom_story_input based on checkbox
    if not use_custom_text:
        custom_story_input = "" # Clear content if unchecked

    if st.button("✨ Generate Content ✨"):
        # Use a spinner while generating
        with st.spinner("Generating content... This might take a while for images on CPU."):
            img, text, messages = generate_content_streamlit(
                prompt_input, text_type, custom_story_input, use_custom_text
            )

        # Display status messages
        for msg in messages:
            if "ERROR" in msg or "blocked" in msg:
                status_area.error(msg)
            elif "WARNING" in msg:
                status_area.warning(msg)
            else:
                status_area.info(msg)

        # Display Outputs
        st.subheader("Generated Content")
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Generated Image")
            if img:
                st.image(img, caption=prompt_input, use_column_width=True)
            else:
                st.info("No image to display.")

        with col2:
            st.write("#### Generated Story/Monologue")
            st.write(text)

    # Optional: Clear status messages after some time or interaction
    # (For a real app, you might want more sophisticated state management)