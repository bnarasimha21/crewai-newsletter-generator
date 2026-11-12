"""Streamlit UI: calls newsletter.pipeline.generate, never touches cwd itself.

Replaces the old version, which imported src.crew directly, wrote
report.html/report.pdf into the process's working directory (so two
concurrent generations stomped on each other), and showed a generic error
message with no way to tell a search failure from a render failure.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from newsletter.errors import RenderFailed, SearchFailed, Timeout
from newsletter.history import list_runs
from newsletter.pipeline import generate

RUNS_DIR = "runs"
HISTORY_DB = Path(RUNS_DIR) / "history.db"

st.title("Newsletter Generator")

with st.form("newsletter_form"):
    topic = st.text_input("Enter a topic", placeholder="e.g., Artificial Intelligence")
    template = st.selectbox("Template", ["classic", "compact"])
    submitted = st.form_submit_button("Generate Newsletter")

if submitted and topic:
    with st.spinner("Researching and writing..."):
        try:
            result = generate(topic, out_dir=RUNS_DIR, template=template, history_db=HISTORY_DB)
        except SearchFailed as exc:
            st.error(f"Could not find enough verifiable sources for '{topic}': {exc}")
        except RenderFailed as exc:
            st.error(f"The newsletter was written but could not be rendered to PDF: {exc}")
        except Timeout as exc:
            st.error(f"Generation timed out before finishing: {exc}")
        else:
            st.success(f"Newsletter generated: {len(result.brief.items)} items.")

            st.subheader("Generated Newsletter")
            with open(result.pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            if hasattr(st, "pdf"):
                st.pdf(pdf_bytes)
            else:
                st.warning("This Streamlit version has no st.pdf; use the download button below.")

            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"{template}-{result.run_dir.name}.pdf",
                mime="application/pdf",
            )

            st.subheader("Sources")
            for item in result.brief.items:
                st.markdown(f"- [{item.title}]({item.source_url}) (source: {item.source_name})")

# Display history from SQLite, not an in-memory session list, so it survives
# a page reload and shows runs started from the CLI too.
st.subheader("Newsletter History")
runs = list_runs(HISTORY_DB)
if runs:
    st.dataframe(
        [
            {
                "Topic": run.topic,
                "Status": run.status,
                "Started": run.started_at,
                "Finished": run.finished_at,
                "Run dir": run.run_dir,
            }
            for run in runs
        ]
    )
else:
    st.caption("No runs yet.")
