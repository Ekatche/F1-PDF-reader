import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path


script_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
relative_path = os.path.join(script_dir, 'functions')
sys.path.insert(1, str(relative_path))

from AppFunctions import load_model, load_data, Vus_df, format_Vus, add_logo

logo_url = './CLBCRCL17.png'
add_logo(logo_url)

######################
####              ####
####  Application ####
####              ####
######################
mut_url = "https://mutalyzer.nl/api/"

st.markdown("<h1 style='text-align:center'> Application page  </h1>", unsafe_allow_html=True)
st.write(
    """
    This page is designed to extract VUS data from Foundation Medecine Report.

    """

)
st.info(
    """
    Process: 
    1. Upload the pdf report  
    2. Launch the preprocessing step 
    3. Then check if the data is correct
    4. Reformat the data (specific for CGI clinics Users)
    """,
    icon="⚙️"
)
st.divider()

classification_model, model, nlp = load_model()

if 'button_1' not in st.session_state:
    st.session_state.button_1 = False
if 'button_2' not in st.session_state:
    st.session_state.button_2 = False
if 'button_3' not in st.session_state:
    st.session_state.button_3 = False

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def cb1():
    st.session_state.button_1 = True


def cb2():
    st.session_state.button_2 = True


def cb3():
    st.session_state.button_3 = True


uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
button1 = st.button("Confirm", on_click=cb1)
st.divider()

if st.session_state.button_1:
    try:
        vus_data = load_data(uploaded_file.read(), classification_model, model, nlp)
        if "vus_df" not in st.session_state:
            st.session_state.vus_df = Vus_df(vus_data)
    except Exception as e:
        st.session_state.button_1 = False
        print(e)
else:
    st.session_state.button_1 = False

st.markdown(
    "<h4 style='text-align: center; color: black; text-decoration: underline;'> Variants of Unknown Significance (VUS) </h4>",
    unsafe_allow_html=True)
if "vus_df" in st.session_state:
    try:
        #########
        # VUS
        #########
        st.info("""check data and correct errors """, icon="ℹ️")
        Vus = Vus_df(vus_data)
        Vus_df = st.data_editor(Vus, num_rows="dynamic", use_container_width=True, key="UniqueVus")
        button = st.button("Reformat the Vus", on_click=cb2)
    except Exception as e:
        del st.session_state['vus_df']
        print("VUS:", e)

else:
    st.info("""please load a file""", icon="ℹ️")

st.divider()
st.markdown(
    "<h4 style='text-align: center; color: black; text-decoration: underline;'> Variants of Unknown Significance (VUS) Reformated </h4>",
    unsafe_allow_html=True)
if st.session_state.button_2:
    try:
        print('formatting Vus')
        final_vus_df = format_Vus(Vus_df, mut_url)
        final_df = st.data_editor(final_vus_df, num_rows="dynamic", use_container_width=True)
    except Exception as e:
        print(f'error formatting vus : {e}')
        st.session_state.button_2 = False
else:
    st.info("""Click button 'reformat the Vus' or check your data """, icon="ℹ️")
