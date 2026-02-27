import os
import time
import base64
import re
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import streamlit as st
from google import genai
from dotenv import load_dotenv

matplotlib.use("Agg")

# Staring
load_dotenv()

# Secret loading 
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("**API Key Missing**: Please add `GEMINI_API_KEY = 'your_key'` to the Streamlit Secrets dashboard.")
    st.stop()

# Create directory for plots if it doesn't exist
PLOT_DIR = "temp_assets"
os.makedirs(PLOT_DIR, exist_ok=True)
PLOT_PATH = os.path.join(PLOT_DIR, "temp_plot.png")

# DATA LOADING
@st.cache_data
def load_titanic_data():
    URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(URL)
    data.columns = [c.lower() for c in data.columns]
    return data

df = load_titanic_data()

# Initializing the Gemini 
client = genai.Client(api_key=GEMINI_API_KEY)

# HELPERS

def extract_python_code(text: str) -> str:
    """Extract code inside markdown blocks."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def compute_numeric_query(prompt: str):
    """Hardcoded stats for high accuracy on common questions."""
    q = prompt.lower()
    if "percentage" in q and "male" in q:
        total = len(df)
        males = df[df['sex'] == 'male'].shape[0]
        return f"{(males/total)*100:.2f}% of passengers were male."
    if "percentage" in q and "female" in q:
        total = len(df)
        females = df[df['sex'] == 'female'].shape[0]
        return f"{(females/total)*100:.2f}% of passengers were female."
    if "survival rate" in q and "class" in q:
        rates = df.groupby('pclass')['survived'].mean().to_dict()
        return "Survival rates by class:\n" + "\n".join([f"Class {k}: {v:.2%}" for k, v in rates.items()])
    if "average age" in q:
        avg_age = df['age'].mean()
        return f"The average age of passengers is {avg_age:.2f} years."
    return None

def query_gemini_with_fallback(prompt: str, models=None, retries=2, delay=2):
    """ Model fallback logic."""
    if models is None:
         models = [
            "gemini-3-flash",
             "gemini-2.5-flash",
             "gemini-1.5-flash"
        ]

    full_prompt = f"""
    You are a Titanic data analyst.
    Use the pandas dataframe 'df' (columns: {', '.join(df.columns)}).
    
    If asked to generate a chart/plot:
    - Provide ONLY Python code (matplotlib/seaborn).
    - Save the plot to '{PLOT_PATH}' using plt.savefig('{PLOT_PATH}').
    - Do NOT use plt.show().
    
    Otherwise, answer briefly in text.
    Question: {prompt}
    """

    last_err = None
    for model_name in models:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt
                )
                return response.text.strip()
            except Exception as e:
                last_err = e
                time.sleep(delay)
                continue
    return f"Error: All models failed. Last error: {last_err}"

# Main

def process_titanic_query(prompt: str):
    """Main entry point for streamlit_app.py"""
    # Clear old plots to avoid stale images
    if os.path.exists(PLOT_PATH):
        try:
            os.remove(PLOT_PATH)
        except:
            pass

    # Checking numeric lookups
    is_viz = any(k in prompt.lower() for k in ["plot", "chart", "graph", "show", "draw"])
    quick_answer = None if is_viz else compute_numeric_query(prompt)
    if quick_answer:
        return quick_answer, None

    text_response = query_gemini_with_fallback(prompt)

    # Handle Plot Execution
    img_b64 = None
    if "```" in text_response or any(kw in text_response for kw in ["plt.", "sns.", "fig ="]):
        try:
            code = extract_python_code(text_response)
            
            plt.clf()
            plt.close('all') 

            # Execution context
            exec_globals = {
                "df": df, "plt": plt, "sns": sns, "pd": pd, 
                "os": os, "PLOT_PATH": PLOT_PATH, "np": __import__('numpy')
            }
            
            exec(code, exec_globals)
            
            if os.path.exists(PLOT_PATH):
                with open(PLOT_PATH, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                text_response = "I have generated the requested visualization."
            else:
                text_response = "I tried to create a plot, but it wasn't saved correctly."
        
        except Exception as e:
            text_response = f"Error during plot generation: {str(e)}"

    return text_response, img_b64



