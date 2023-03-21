import json
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
#import summarize.ipynp

option_menu(
    menu_title=None,
    options=['Home', 'Our Porject', 'About US'],
    icons=['house', 'book', 'envelop'],

)


st.title('Text summurization ')


hide_st_style = """ 
         <style>
             footer{ visiblity: hidden}
             header{ visibility: hidden}
             #MainMenu {visibility: hidden}
         </style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)


text_input = st.text_input("Enter some text:")


with open('fine_tuned_model.pkl', 'rb') as f:
    model = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")


def generate_summary(text, max_length=200, min_length=30):
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = summarizer(text, max_length=max_length,
                         min_length=min_length, do_sample=True)

    return summary[0]["summary_text"]


if st.button("Summarize"):
    summary = generate_summary(text_input)
    st.write(f"Summary: {summary}")
