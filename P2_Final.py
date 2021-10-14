# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:18:22 2021

@author: Admin
"""
import nltk
nltk.download('wordnet')
import pandas as pd
import warnings
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
import re
from rake_nltk import Rake
import pickle
import streamlit as st
import numpy as np
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# Warnings ignore 
warnings.filterwarnings(action='ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)

# loading the trained model
pickle_in = open('nlp_model1.pkl', 'rb') 
model = pickle.load(pickle_in)

pickle_in = open('vectorizer.pkl', 'rb') 
vectorizer = pickle.load(pickle_in)
# Title of the application 
st.title('Email classification Tool\n', )
st.header("Email classification into Abusive & Non Abusive")
st.subheader("Enter the email content you want to check the class for:")

input_text = st.text_area("Enter email content here", height=50)


# Sidebar options
option = st.sidebar.selectbox('Navigation',['Class detection','Keywords','Word Cloud'])
st.set_option('deprecation.showfileUploaderEncoding', False)
if option == "Class detection":
    
    
    
    if st.button("Predict Email class"):
        st.write("Number of words in Email:", len(input_text.split()))
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        pickle_in = open('nlp_model1.pkl', 'rb') 
        model = pickle.load(pickle_in)
        pickle_in = open('vectorizer.pkl', 'rb') 
        vectorizer = pickle.load(pickle_in)
        transformed_input = vectorizer.transform([text])
        names_features = vectorizer.get_feature_names()
        dense = transformed_input.todense()
        denselist = dense.tolist()
        tf= pd.DataFrame(denselist, columns = names_features)
        prediction = model.predict(tf)
	
        
        if model.predict(tf) == 'Abusive':
            st.write("Input Email content class is : Abusive as it contains some bad words in it.")
        elif model.predict(tf) == 'Non Abusive':
            st.write("Input Email content class is : Non-Abusive as it doesn't contain any bad words in it.")
        else:
            st.write("Input email is normal.😐")
         
elif option == "Keywords":
    st.header("Keywords")
    if st.button("Keywords"):
        
        r=Rake(language='english')
        r.extract_keywords_from_text(input_text)
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Display the important phrases
        st.write("These are the **keywords** causing the above sentiment:")
        for i, p in enumerate(phrases):
            st.write(i+1, p)
elif option == "Word Cloud":
    st.header("Word cloud")
    if st.button("Generate Wordcloud"):
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        wordcloud = WordCloud().generate(text)
        plt.figure(figsize=(40, 30))
        plt.imshow(wordcloud) 
        plt.axis("off")
        
        st.pyplot()