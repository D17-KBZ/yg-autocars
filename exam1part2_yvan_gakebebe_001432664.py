import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(layout="wide")

# Title and description
st.title("Car Price Predictor - Data Analysis")
st.markdown("This app explores the relationship between car characteristics and their impact on price.")

# 1. Load data
st.header("1. Load and Inspect the Dataset")
url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/main/CleanedAutomobile.csv'
df = pd.read_csv(url)
st.dataframe(df.head())

# 2. Analyzing Individual Feature Patterns Using Visualization
st.header("2. Feature Correlations and Visualizations")
st.markdown("### What is the data type of 'peak-rpm'?")
st.write(df.dtypes['peak-rpm'])

st.markdown("### Correlation Matrix")
selected_columns = df[['bore', 'stroke', 'compression-ratio', 'horsepower']]
correlation_matrix = selected_columns.corr()
st.dataframe(correlation_matrix)

# Scatterplots
st.subheader("Regression Plots")
fig1 = sns.regplot(x="engine-size", y="price", data=df)
st.pyplot(fig1.figure)

fig2 = sns.regplot(x="highway-mpg", y="price", data=df)
st.pyplot(fig2.figure)

fig3 = sns.regplot(x="peak-rpm", y="price", data=df)
st.pyplot(fig3.figure)

# Stroke vs price correlation
st.subheader("Correlation: Stroke vs. Price")
st.write(df[['stroke', 'price']].corr())

fig4 = sns.regplot(x="stroke", y="price", data=df)
st.pyplot(fig4.figure)

# Categorical Variables
st.header("Boxplots of Categorical Variables vs Price")
fig5 = sns.boxplot(x="body-style", y="price", data=df)
st.pyplot(fig5.figure)

fig6 = sns.boxplot(x="engine-location", y="price", data=df)
st.pyplot(fig6.figure)

fig7 = sns.boxplot(x="drive-wheels", y="price", data=df)
st.pyplot(fig7.figure)

# 3. Descriptive Statistics
st.header("3. Descriptive Statistical Analysis")
st.write(df.describe())
st.write(df.describe(include='object'))

# Value Counts
st.subheader("Drive Wheels Count")
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
st.write(drive_wheels_counts)

st.subheader("Engine Location Count")
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
st.write(engine_loc_counts)

# 4. Basics of Grouping
st.header("4. Grouping and Pivot Tables")

# Select relevant columns
df_group_one = df[['drive-wheels', 'body-style', 'price']]

# Only average numeric columns (price), keep 'drive-wheels' for grouping
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False)[['price']].mean()

# Display result
st.write(df_group_one)


grouped_test1 = df[['drive-wheels','body-style','price']].groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)
st.dataframe(grouped_pivot)

st.markdown("### Heatmap of Drive-Wheels and Body-Style vs Price")
fig8, ax = plt.subplots()
sns.heatmap(grouped_pivot, annot=True, fmt=".0f", cmap="RdBu", ax=ax)
st.pyplot(fig8)

# 5. Correlation and Causation
st.header("5. Pearson Correlation Coefficients with Price")
st.write(df.corr(numeric_only=True)['price'].sort_values(ascending=False))

st.markdown("### Pearson Correlation Test (wheel-base vs price)")
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
st.write(f"**Pearson Coefficient**: {pearson_coef:.3f}")
st.write(f"**P-value**: {p_value:.3e}")
