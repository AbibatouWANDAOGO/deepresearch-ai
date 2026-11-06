"""Streamlit interface. Phase 1: shows the configuration status only.

    streamlit run frontend/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

# Streamlit puts this file's folder on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import __version__
from app.config import Settings

st.set_page_config(page_title="DeepResearch AI", page_icon="🔎", layout="centered")

st.title("DeepResearch AI")
st.caption(f"Version {__version__}")
st.write(
    "An autonomous research agent: it plans a research question, searches the web, "
    "collects pages and PDFs, analyzes them with retrieval-augmented generation, "
    "fact-checks every important claim and writes a report with traceable citations."
)

st.subheader("Configuration")
try:
    settings = Settings()
except ValidationError as exc:
    st.error("The configuration is invalid. Fix your .env file and reload the page.")
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "settings"
        st.write(f"- `{location}`: {error['msg']}")
else:
    st.success("Configuration loaded.")
    st.table({"Setting": list(settings.summary()), "Value": list(settings.summary().values())})
    if not settings.allow_paid_apis:
        st.info("Paid APIs are disabled: the app only uses free, local services.")

st.subheader("Research")
st.info("Research is not available yet: it will be enabled as the next phases are built.")
