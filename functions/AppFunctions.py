# F1 streamlit page 
import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import tempfile
import re
import math
from pathlib import Path
import glob
import spacy
import warnings

### ADD logo

import base64

warnings.filterwarnings('ignore')
from stqdm import stqdm
from doctr.models import ocr_predictor
from pdf2image import convert_from_bytes
import requests
import os
from pathlib import Path

script_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
relative_path_class = os.path.join(script_dir, 'models', 'final_3class_classifier.h5')
relative_path_spacy = os.path.join(script_dir, 'models', 'model-best')


@st.cache_resource()
def load_ocr_model():
    """
    Load classification model and OCR model and Spacy Model
      """

    model = ocr_predictor(det_arch='linknet_resnet34', reco_arch='crnn_vgg16_bn', pretrained=True,
                          assume_straight_pages=True,
                          preserve_aspect_ratio=True
                          )
    return model

@st.cache_resource()
def load_classification_model():
    """
    Load classification model and OCR model and Spacy Model
    """
    classification_model = tf.keras.models.load_model(relative_path_class)


    return classification_model

@st.cache_resource()
def load_spacy_model():
    """
    Load classification model and OCR model and Spacy Model
    """
    nlp = spacy.load(relative_path_spacy)

    return nlp


def replace(match):
    return 'NM_' + match.group(2).replace(' ', '_')


@st.cache_data(show_spinner="Extracting data from pdf ...")
def load_data(uploaded_file, _classification_model, _model, _nlp):
    path = tempfile.mkdtemp()
    images_from_path = convert_from_bytes(uploaded_file, output_folder=path, fmt="png")

    #######################################################
    ############### REGEX PATTERN #########################
    #######################################################

    # pattern to detect NM_003482.4
    patern1 = r'NM[_\s]*\d+(?:\.\d+)?'
    # pattern to detect c.1039C>T
    patern2 = r'[Cc]\.\d+(?:_\d+)?[ACGTacgt\s]*>[ACGTacgt]+|[Cc]\.\d+\s\d+\w+|[Cc]\.\d+(?:\s+\d+)+[ACGTacgt]*\d|[Cc]\.\d+(?:_\d+)?delins[ACGTacgt\s]+[ACGTacgt]+|[Cc]\.\d+[ACGTacgt]*[ACGTacgt]+|[Cc]\.\d+\S\d+\w+'
    # pattern to chr11:118377130 
    patern3 = r'(chr[XY\d]+:\d+|chr[XY\d]+\d+-\d+|chr\d+)'

    #######################################################
    ############### Extract Vus #################
    #######################################################

    formatted_lines = []
    temp_list = []
    target_size = (224, 224)

    for image in stqdm(glob.glob(path + "/" + "*")):
        img = cv2.imread(image)
        resized_image = cv2.resize(img, target_size)
        predictions = _classification_model.predict(np.expand_dims(resized_image, axis=0))
        pred = tf.math.argmax(predictions[0])

        if pred == 2:
            image_np = np.array(img)
            new_width = image_np.shape[1] * 2
            new_height = image_np.shape[0] * 2
            resized_image = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            result = _model([resized_image])
            text = result.render()
            if "APPENDIX Variants of Unknown" in text:
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    line = line.replace('/', '7')
                    doc = _nlp(line)
                    if len(doc.ents) > 1:
                        if doc.ents[0].label_ == 'REFDEQ':
                            gene = doc.ents[0].text
                            refseq = re.sub(r'NM\s*(_*\s*)(\d+\.\d+)', replace, gene.replace('/', '7'))
                            temp_list.append(refseq)

                        if doc.ents[0].label_ == 'CCHANGE':
                            mutation = doc.ents[0].text
                            cchange = mutation.replace(' ', '_')
                            temp_list.append(cchange)

                        if doc.ents[0].label_ == 'PCHANGE':
                            pchange = doc.ents[0].text
                            formated_pchange = re.sub(r",", "", pchange)
                            temp_list.append(formated_pchange)

                        if doc.ents[1].label_ == 'REFDEQ':
                            gene = doc.ents[1].text
                            refseq = re.sub(r'NM\s*(_*\s*)(\d+\.\d+)', replace, gene.replace('/', '7'))
                            temp_list.append(refseq)

                        if doc.ents[1].label_ == 'CCHANGE':
                            mutation = doc.ents[1].text
                            cchange = mutation.replace(' ', '_')
                            temp_list.append(cchange)

                        if doc.ents[1].label_ == 'PCHANGE':
                            pchange = doc.ents[1].text
                            formated_pchange = re.sub(r",", "", pchange)
                            temp_list.append(formated_pchange)

                        if doc.ents[1].label_ == 'CHR':
                            mutation = doc.ents[1].text
                            cchange = mutation.replace(' ', '_')
                            temp_list.append(cchange)

                    if len(doc.ents) == 1:
                        if doc.ents[0].label_ == 'CHR':
                            chrom = doc.ents[0].text
                            if 'chr' in chrom:
                                formated_chrom = re.sub(r"(chr\d{2})(\d+)(-\d+)", r"\1:\2\3", chrom)
                                temp_list.append(formated_chrom)

                            if re.search(patern2, chrom) and re.match(patern1, chrom):
                                refseq = re.findall(patern1, chrom)[0]
                                cchange = re.findall(patern2, chrom)[0]
                                formted_refseq = refseq.replace('.', '_')
                                temp_list.append(formted_refseq)
                                temp_list.append(cchange)

                        if doc.ents[0].label_ == 'PCHANGE':
                            pchange = doc.ents[0].text
                            formated_pchange = re.sub(r",", "", pchange)
                            temp_list.append(formated_pchange)

                        if doc.ents[0].label_ == "REFDEQ":
                            gene = doc.ents[0].text
                            refseq = re.sub(r'NM\s*(_*\s*)(\d+\.\d+)', replace, gene.replace('/', '7'))
                            temp_list.append(refseq)

                        if doc.ents[0].label_ == 'CCHANGE':
                            mutation = doc.ents[0].text
                            cchange = mutation.replace(' ', '_')
                            temp_list.append(cchange)
                    if len(doc.ents) < 1:
                        text = line
                        if re.search(patern2, text) and re.match(patern1, text):
                            refseq = re.findall(patern1, text)[0]
                            cchange = re.findall(patern2, text)[0]
                            formted_refseq = refseq.replace('.', '_')
                            temp_list.append(formted_refseq)
                            temp_list.append(cchange)

                    if len(temp_list) > 3:  # Combine information into a single list when at least three elements are present
                        formatted_lines.append(temp_list)
                        temp_list = []
        else:
            pass

    return formatted_lines


@st.cache_data()
def Vus_df(formatedLines):
    Vus = pd.DataFrame(formatedLines, columns=["RefSeq", "Mutation_loc", 'Protein_mut', "Chromosome_pos"])
    vus_copy = Vus.copy()
    return vus_copy


@st.cache_data(show_spinner="Formating Vus ...")
def format_Vus(Vus, mut_url):
    Vus['Mutation_loc'] = Vus["Mutation_loc"].apply(lambda a: a.split(".")[0].lower() + '.' + a.split(".")[1])
    Vus['Protein_mut'] = Vus["Protein_mut"].apply(lambda a: a.split(".")[0].lower() + '.' + a.split(".")[1])
    try:
        esembl = True
        Vus['ensemblID'] = Vus.RefSeq.apply(
            lambda x: requests.get(f'{mut_url}/related_references/{x.strip()}').json()['ensembl'][1]['id'])
    except Exception as e:
        esembl = False
        print(e)

    if esembl:
        Vus['Transcript_variant'] = Vus.apply(lambda x: x['ensemblID'] + ":" + x['Mutation_loc'].strip(), axis=1)
    else:
        print(f'problem during emsemblId loading')

    Vus_copy = Vus[['RefSeq', 'Transcript_variant']].copy()
    print("done formating vus")
    return Vus_copy


@st.cache_resource()
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(
        png_file,
        background_position="50% 10%",
        margin_top="10%",
        image_width="60%",
        image_height="",
):
    binary_string = get_base64_of_bin_file(png_file)
    return """
            <style>
                [data-testid="stSidebarNav"] {
                    background-image: url("data:image/png;base64,%s");
                    background-repeat: no-repeat;
                    background-position: %s;
                    margin-top: %s;
                    background-size: %s %s;
                }
            </style>
            """ % (
        binary_string,
        background_position,
        margin_top,
        image_width,
        image_height,
    )


def add_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )


