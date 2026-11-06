"""Native extra page: no changes to the existing entrypoint or its state."""
from pathlib import Path
import sys

import streamlit as st

st.set_page_config(page_title='PyC | Archivos mainframe', page_icon='📁', layout='wide')
# Streamlit pages use the parent's project modules, not uploaded executable code.
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from mainframe_ui import render

render()
