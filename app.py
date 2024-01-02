import streamlit as st
import sys
import os
from pathlib import Path

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
relative_path = os.path.join(script_dir, 'functions')
sys.path.insert(1, str(relative_path))

from AppFunctions import add_logo

# This page works like a main entry page of the streamlit app

logo_url = './CLBCRCL17.png'

add_logo(logo_url)
st.markdown(
    '<h1 style = "text-align:center"> AI based tool to extract information from Foundation Medecine PDF reports </h1>',
    unsafe_allow_html=True)

tab1, tab2 = st.tabs(['Introduction', 'Process'])

with tab1:
    st.markdown('''
    <p style="text-align: justify"> This application was developed at the Léon Bérard Center (CLB) as part of the European project CGI (Cancer Genome Interpreter) clinics. 
    <a href="https://www.cgiclinics.eu/" > The Cancer Genome Interpreter tool</a>  is designed to support the identification of tumor alterations that drive the 
    disease and/or which may be therapeutically actionable.
        
    In order to analyze a maximum amount of data with this tool, we needed to access data present in several PDF reports, 
    such as those provided by Foundation Medicine. Since we did not have access to the database, we developed this tool 
    to directly extract data from variants of unknown significance, allowing us to analyze them with the CGI clinics tool.
    </p>''', unsafe_allow_html=True)

with tab2:
    st.subheader('This is how this app works')
    st.markdown("""
    ### 1 - Image classification

When you upload your PDF, it is initially converted into an image. Subsequently, it undergoes processing through an
image classification model, which filters the report and retains only the images containing the relevant data that we
aim to extract.

### 2 - Image conversion

Once the images of interest are chosen, we utilize OCR (Optical Character Recognition) to convert these images into
text. For those who may not be familiar with the term, OCR is a technology that enables the extraction of text from
images. In the context of this application, we have employed [docTR OCR](https://github.com/mindee/doctr), developed by
Mindee, to carry out the OCR
process.

### 3 - Information retrieval

In this stage, we applied a customized Spacy Named Entity Recognition (NER) model, complemented by regular expression (
regex)
patterns, to extract the pertinent information. For those not familiar with the term, NER is a technology used to
identify and classify named entities, such as names, locations, within a given text.

### 4 - Reorganize the Data Layout

Finally, we reformat the data into a tabular format, enabling users to analyze it conveniently
using [CGI clinics](https://www.cgiclinics.eu/).
    """)
    st.image("./data_extraction_workflow.png")
