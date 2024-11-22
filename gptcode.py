import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create containers
header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

# Header
with header:
    st.title("Kashti ki App")
    st.text("In this Project we will work on Kashti Data")

# Data Set
with data_sets:
    st.header("Kashti Doob gayi")
    st.text("We will work on Titanic Data Set")
    # Load data set
    df = sns.load_dataset("titanic")
    df = df.dropna()
    st.write(df.head())
    st.subheader("Gender k hisaab se ploting")
    st.bar_chart(df['sex'].value_counts())
    st.subheader("Class k hisaab se farq by bar chart")
    st.bar_chart(df['class'].value_counts())
    st.subheader("Age k hisaab se bar chart")
    st.bar_chart(df['age'].sample(10))
    
# Features
with features:
    st.header("Features of Kashti:")
    st.text("Features ziada kiun hote hain:")
    st.markdown('1.**Feature 1** it will tell us about features of this app')
    st.markdown('2.**Feature 2** it will tell us about features of this app')
    st.markdown('3.**Feature 3** it will tell us about features of this app')

# Model Training
with model_training:
    st.header("Kashti walon ka kiya bana? Model Training")
    st.text("is mein hum apne parameters ko kam ya ziada karen ge")
    # Create columns
    input, display = st.columns(2)
    
    # Slider for max_depth
    max_depth = input.slider("Tree depth:", min_value=10, max_value=100, value=20, step=5)
    
    # Selectbox for n_estimators
    n_estimators = input.selectbox("Number of trees:", options=[50, 100, 200, 300])
    
    # Show available columns
    input.write("Available features:")
    input.write(df.columns)
    
    # Input feature
    input_features = input.text_input("Enter a feature to use (e.g., 'age')", value='age')
    
    # Validate input feature
    if input_features not in df.columns:
        st.error(f"'{input_features}' is not a valid feature. Please select from the available features.")
    else:
        # Define X and y
        X = df[[input_features]]
        y = df['fare']
        
        # Train model
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
        model.fit(X, y)
        pred = model.predict(X)
        
        # Display metrics
        display.subheader("Model Performance Metrics:")
        display.write("Mean Absolute Error: ", mean_absolute_error(y, pred))
        display.write("Mean Squared Error: ", mean_squared_error(y, pred))
        display.write("R2 Score: ", r2_score(y, pred))
