# import libraries
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web Title
st.markdown('''
# **Exploratory Data Analysis Web Application**
This application is developed by me after watching video of Codianics Called **EDA App**
''')
# How to upload a file from PC
with st.sidebar.header("Upload a Data Set (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](https://githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)")
# Profiling report for pandas
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input Data Frame**')
    st.write(df)
    st.write('---')
    st.header('**Profiling report with Pandas**')
    st_profile_report(pr)
else:
    st.info("Awaiting for the file to be uploaded.")
    if st.button('Press to use example data'):
        # example data set
        @st.cache_data
        def load_data():
            a = pd.DataFrame(np.random.rand(100,5),
                             columns=['Age','Banana','Codianic','Deutchland','Ear'])
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input Data Frame**')
        st.write(df)
        st.write('---')
        st.header('**Profiling report with Pandas**')
        st_profile_report(pr)
    
    
