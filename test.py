import streamlit as st
import seaborn as sns

st.header("This is first line code")
st.text("Baba G bohat maza a raha hai")
st.header("Ye to kamal ho gaya Baba G")

df = sns.load_dataset('iris')
st.write(df.head(10))

st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])