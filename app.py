import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import base64

# Add the root directory to Python path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from src.crew import CrewExperiment

def display_pdf(pdf_path):
    """Display PDF in Streamlit"""
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

PDF_PATH = "report.pdf"

def process_topic(topic: str):
    inputs = {'topic': topic}
    # Remove any stale PDF so a failed conversion is not mistaken for success
    if os.path.exists(PDF_PATH):
        os.remove(PDF_PATH)
    with st.spinner('Generating newsletter...'):
        CrewExperiment().crew().kickoff(inputs=inputs)
    return PDF_PATH if os.path.exists(PDF_PATH) else None

# Initialise history on first load (session_state.history is read below on every run)
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Topic", "Generated At"])

# Streamlit UI
st.title("Newsletter Generator")

# Input section
with st.form("newsletter_form"):
    topic = st.text_input("Enter a topic", placeholder="e.g., Artificial Intelligence")
    submitted = st.form_submit_button("Generate Newsletter")

# Process form submission
if submitted and topic:
    pdf_file = process_topic(topic)
    if not pdf_file:
        st.error("The crew finished but no report.pdf was produced. "
                 "Check that wkhtmltopdf is installed and the API keys are set.")
    else:
        st.session_state.history = pd.concat(
            [st.session_state.history,
             pd.DataFrame([{"Topic": topic,
                            "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])],
            ignore_index=True)
        st.success("Newsletter generated successfully!")
        
        # Display PDF
        st.subheader("Generated Newsletter")
        display_pdf(pdf_file)
        
        # Download button
        with open(pdf_file, "rb") as file:
            st.download_button(
                label="Download PDF",
                data=file,
                file_name=pdf_file,
                mime="application/pdf"
            )

# Display history
if not st.session_state.history.empty:
    st.subheader("Newsletter History")
    st.dataframe(st.session_state.history)
