import streamlit as st
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
st.set_page_config(page_title="Sentiment Analyzer")
st.title("Sentiment Analyzer")
st.write("Enter some text below to analyze its sentiment.")

text = st.text_area("Your text here")

analyze = st.button("Analyze")

if analyze and not text:
    st.warning("Please enter some text first.")
elif analyze or (text and "\n" not in text):
    result = sentiment_pipeline(text)
    label = result[0]["label"]
    score = result[0]["score"]
    if label == "LABEL_0":
        st.markdown(f"<p style='color: red'> Negative</p>", unsafe_allow_html=True)
    elif label == "LABEL_1":
        st.markdown(f"<p style='color: gray'> Neutral</p>", unsafe_allow_html=True)
    elif label == "LABEL_2":
        st.markdown(f"<p style='color: green'> Positive</p>", unsafe_allow_html=True)
    st.write(f"Confidence: **{score:.2%}**")
