import json
import pickle
import streamlit as st

import pandas as pd
import numpy as np
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
#import summarize.ipynp




st.title('Text summurization ')







text_input = st.text_input("Enter some text:")

model = AutoModelForSeq2SeqLM.from_pretrained("SmonF/YTFineTunePegasus")
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")


def generate_summary(text, max_length=200, min_length=30):
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = summarizer(text, max_length=max_length,
                         min_length=min_length, do_sample=True)

    return summary[0]["summary_text"]


if st.button("Summarize"):
    summary = generate_summary(text_input)
    st.write(f"Summary: {summary}")
