"""Additive read-only comparator; the original relational app is unchanged."""
import sys
from pathlib import Path

import streamlit as st

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

st.set_page_config(page_title='Comparador A/B · PyC', page_icon='📊', layout='wide')
from comparison_ui import render_comparator

render_comparator()
