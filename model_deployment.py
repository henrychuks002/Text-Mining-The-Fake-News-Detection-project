#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:07:54 2021

@author: Henry Chuks
"""

import streamlit as st
import pandas as pd
import data_conversion as cd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
# =============================================================================
# import os
# import sys
# =============================================================================

header = st.beta_container()
dataset = st.beta_container()
model_testing = st.beta_container()

with open("pac.pk1","rb") as f:
    classifier = pickle.load(f)

with header:
    st.title("Text classification: Fake News Detection model deployment - Streamlit application")
    st.markdown('**What is Text Classification?**')
    st .write('''
             Text classification also known as text tagging or text categorization is\n \
             the process of categorizing text into organized groups. By using Natural Language Processing (NLP),\n \
             text classifiers can automatically analyze text and then assign a set of pre-defined tags or categories based on its content.
             ''')
    st.write('''
             This project is about the fake news detection, a model that classifies news text or content into real or fake. It takes a particular
             each text sample from the data and classifies it into one the two binary class **real** or **fake**, after the text transformation though,
             and text cleaning, removing punctuations, tokenizing words, vectorizing words via count vectorizer and finally transforming them via tfidf
             transformer, and then passing the already made data through the learning algorithms.
             '''
            )
       
with dataset:
    st.header('Overview of the train data')
    st.write('''
             The dataset below shows a small sample of what the train data looks like. It contains only four columns, the news title, the
             news content, date, new subject and then the category as the class. The data was gotten from https://archive.ics.uci.edu/ml/datasets.php
             Not all of these data features as mentioned will be used to build the model though, the only required feature for this project is the text column.
             All other column will be dropped, as there are reasons why.
             ''')
    dataset = pd.read_csv('News_data.csv')
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    st.write(dataset.head())
    
with model_testing:
    st.header('Model testing with data')
    st.write('Here is to upload any test data and show prediction result')
    file = st.file_uploader("Upload Data File", type='csv')
    show_file = st.empty()
    
    if not file:
        show_file.info("Please Upload a data file: {}".format(''.join("csv")))
        
    content = file.getvalue()
    df = pd.read_csv(file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=0)
    df = df.iloc[0:100]
    st.dataframe(df.head(10))
    y = df['category']
    new_y = cd.target(df)
    st.write('''
             A small sample of your test data is shown above, in general your test data contains {} columns and {} rows. Click the
             **predict** button below to get the results of your test data
             '''.format(df.shape[1], df.shape[0]))
    var = cd.convert_data(df)
    st.write('Transfromed dataset shape: {}'.format(var.shape))
    if st.button("Predict"):
        prediction = classifier.predict(var)
        predicted = prediction.map({0:'Real', 1:"Fake"})
        actual_values = list(np.array(y))
        predicted_values = list(predicted)
        list_comb = np.array([[a,b] for (a,b) in zip(actual_values, predicted_values)])
        st.write("Your test data results are")
        st.write(pd.DataFrame(data=list_comb, columns=['Actual values', 'Predicted values']))
        st.write('Accuracy score: ',accuracy_score(list(new_y.values), prediction))
        
    