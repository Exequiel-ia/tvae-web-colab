"""Saved relational models; separate from the read-only A/B comparator."""
import sys
from pathlib import Path
import streamlit as st

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
st.set_page_config(page_title='Biblioteca de modelos · PyC', page_icon='💾', layout='wide')
from model_library_ui import render_library
render_library()
