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

# Use Agg backend for headless environments like Streamlit Cloud
matplotlib.use("Agg")

# --- Load environment variables / Secrets ---
load_dotenv()
# Prioritize Streamlit Secrets for cloud deployment, fallback to .env for local
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please add it to Streamlit Secrets.")
    st.stop()

# --- Assets setup ---
# Streamlit Cloud needs a writable directory
STATIC_DIR = "temp_assets"
os.makedirs(STATIC_DIR, exist_ok=True)
PLOT_PATH = os.path.join(STATIC_DIR, "temp_plot.png")

# --- Load Titanic dataset ---
@st.cache_data
def load_data():
    URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(URL)
    data.columns = [c.lower() for c in data.columns]
    return data

df = load_data()

# --- Initialize Gemini client ---
client = genai.Client(api_key=GEMINI_API_KEY)

# --- Helpers ---
def extract_python_code(text: str) -> str:
    """Extract code inside ```python ... ``` or ``` ... ``` blocks."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def compute_numeric_query(prompt: str):
    """Compute numeric/statistics answers from the dataset."""
    prompt_lower = prompt.lower()
    if "percentage" in prompt_lower and "male" in prompt_lower:
        total = len(df)
        males = df[df['sex'] == 'male'].shape[0]
        return f"{(males/total)*100:.2f}% of passengers were male"
    if "percentage" in prompt_lower and "female" in prompt_lower:
        total = len(df)
        females = df[df['sex'] == 'female'].shape[0]
        return f"{(females/total)*100:.2f}% of passengers were female"
    if "survival rate" in prompt_lower and "class" in prompt_lower:
        rates = df.groupby('pclass')['survived'].mean().to_dict()
        return "Survival rates by class:\n" + "\n".join([f"Class {k}: {v:.2%}" for k,v in rates.items()])
    if "average age" in prompt_lower:
        avg_age = df['age'].mean()
        return f"The average age of passengers is {avg_age:.2f} years"
    return None

def query_gemini_with_fallback(prompt: str, models=None, retries=2, delay=2):
    """Call Gemini models with fallback logic (Models kept as requested)."""
    if models is None:
        models = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-1.5-flash"]

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

    last_exception = None
    for model in models:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=full_prompt
                )
                return response.text.strip()
            except Exception as e:
                last_exception = e
                time.sleep(delay)
                continue
    return f"Error: All models failed. Last error: {last_exception}"

def process_titanic_query(prompt: str):
    """Main logic function called by streamlit_app.py"""
    # Clean the old plot
    if os.path.exists(PLOT_PATH):
        try:
            os.remove(PLOT_PATH)
        except Exception:
            pass

    # Checking numeric queries first
    is_plot = any(k in prompt.lower() for k in ["plot", "chart", "graph", "show", "draw"])
    numeric_answer = None if is_plot else compute_numeric_query(prompt)
    
    if numeric_answer:
        return numeric_answer, None

    # Calling Gemini with fallback
    text_response = query_gemini_with_fallback(prompt)

    # Python plotting code execution
    img_base64 = None
    if "```" in text_response or any(kw in text_response for kw in ["plt.", "sns.", "fig ="]):
        try:
            code = extract_python_code(text_response)
            
            plt.clf()
            plt.close('all') 

            exec_globals = {
                "df": df, 
                "plt": plt, 
                "sns": sns, 
                "pd": pd, 
                "os": os,
                "PLOT_PATH": PLOT_PATH,
                "np": __import__('numpy')
            }
            
            exec(code, exec_globals)
            
            if os.path.exists(PLOT_PATH):
                with open(PLOT_PATH, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                text_response = "I have generated the requested visualization."
            else:
                text_response = "I tried to create a plot, but the file wasn't saved correctly."
        
        except Exception as e:
            text_response = f"I encountered an error while generating the plot: {str(e)}"

    return text_response, img_base64
