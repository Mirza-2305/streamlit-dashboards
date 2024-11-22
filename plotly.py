# import libraries
import streamlit as st
import plotly.express as px
import pandas as pd

# load data set from plotly
st.title('To make an App by combining Plotly and Streamlit')
df = px.data.gapminder()
st.write(df)
st.write(df.head())
st.write(df.columns)

# Summary State
st.write(df.describe())

# data management
year_option = df['year'].unique().tolist()

year = st.selectbox("Which year you want to ploy?", year_option, 0)
#df = df[df['year']== year]

# Plotting
fig = px.scatter(df, x='gdpPercap', y='lifeExp', size='pop', color='country', hover_name='country',
                 log_x=True, size_max=55, range_x=[100,100000], range_y=[20,90],
                 animation_frame='year', animation_group='country')
# to change the size of fig/plot
fig.update_layout(width=1000, height=800)
st.write(fig)
