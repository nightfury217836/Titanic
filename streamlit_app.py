import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from backend import process_query 

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Insights",
    page_icon="🚢",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 12px; border: 1px solid #e9ecef; }
    /* Hide the top streamlit menu and footer for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("⚓ Titanic Data Assistant")
st.markdown("Ask questions about passenger demographics, survival rates, or request custom charts.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            try:
                img_bytes = base64.b64decode(msg["image"])
                st.image(
                    Image.open(BytesIO(img_bytes)),
                    use_container_width=True
                )
            except Exception:
                pass

# --- Chat Input & Logic ---
if prompt := st.chat_input("Ask about Titanic data..."):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Call the backend logic directly
                bot_response, img_data = process_query(prompt)

                # Render Text Response
                st.markdown(bot_response)

                # Render Image if present
                if img_data:
                    img_bytes = base64.b64decode(img_data)
                    st.image(
                        Image.open(BytesIO(img_bytes)),
                        use_container_width=True,
                        caption="Analysis Result"
                    )

                # 3. Store in History
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_response,
                    "image": img_data
                })

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")