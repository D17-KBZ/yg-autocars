# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set up the page
st.set_page_config(page_title="Car Price Analysis", layout="wide")
st.title("Car Price Analysis")
st.markdown("""
<h3>What are the main characteristics that have the most impact on the car price?</h3>
""", unsafe_allow_html=True)

# Load data from GitHub
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/main/CleanedAutomobile.csv'
    df = pd.read_csv(url)
    return df

df = load_data()

# Section 1: Show Data
st.header("1. Dataset Overview")
st.write("First few rows of the dataset:")
st.dataframe(df.head())

# Section 2: Data Types
st.header("2. Data Types")
st.write("Data types of each column:")
st.write(df.dtypes.astype(str))

# Section 3: Correlation Analysis
st.header("3. Correlation Analysis")

# Question 1: Data type of 'peak-rpm'
st.subheader("Question 1: Data Type of 'peak-rpm'")
st.write(f"Data type of 'peak-rpm': **{df['peak-rpm'].dtype}**")

# Question 2: Correlation between selected features
st.subheader("Question 2: Correlation Matrix")
selected_cols = ['bore', 'stroke', 'compression-ratio', 'horsepower']
st.write("Correlation matrix for bore, stroke, compression-ratio, horsepower:")
corr_matrix = df[selected_cols].corr()
st.dataframe(corr_matrix)

# Section 4: Visualizations
st.header("4. Feature Visualization")

# Engine Size vs Price
st.subheader("Engine Size vs Price")
fig, ax = plt.subplots()
sns.regplot(x="engine-size", y="price", data=df, ax=ax)
ax.set_ylim(0,)
st.pyplot(fig)
plt.clf()

# Highway MPG vs Price
st.subheader("Highway MPG vs Price")
fig, ax


