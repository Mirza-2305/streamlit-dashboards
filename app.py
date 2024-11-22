import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# make containers

header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()
# make header
with header:
    st.title("Kashti ki App")
    st.text("In this Project we will work on Kashti Data")

with data_sets:
    st.header("Kashti Doob gayi")
    st.text("We will work on Titanic Data Set")
    # load data set
    df = sns.load_dataset("titanic")
    df = df.dropna()
    st.write(df.head())
    st.subheader("Gender k hisaab se ploting")
    st.bar_chart(df['sex'].value_counts())
    # other plot
    st.subheader("Class k hisaab se farq. by bar chart")
    st.bar_chart(df['class'].value_counts())
    # another bar plot of age by sample or by using head
    st.subheader("Age k hisaab se bar chart")
    st.bar_chart(df['age'].sample(10))
    
with features:
    st.header("Features of Kashti:")
    st.text("Features ziada kiun hote hain:")
    st.markdown('1.**Feature 1** it will tell us about features of this app')
    st.markdown('2.**Feature 2** it will tell us about features of this app')
    st.markdown('3.**Feature 3** it will tell us about features of this app')
    
with model_training:
    st.header("Kashti walon ka kiya bana? Model Training")
    st.text("is mein hum apne parameters ko kam ya ziada karen ge")
    # making columns
    input, display = st.columns(2)
    
    # pehley column mein hamare selection points hon
    max_depth = input.slider("How many people do you know?", min_value=10, max_value=100, value=20, step=5)
    
    # n_estimators
    n_estimators = input.selectbox("How many trees should be there in RF?", options=[50,100,200,300,'No Limit'])

    # adding list of features
    input.write("Available features:")
    input.write(df.columns)

    # Input features from User
    input_features = input.text_input("Enter a feature to use (e.g., 'age')", value='age')

    # Machine learning model
    #model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    # define X and y
    #X = df[[input_features]]
    #y = df[['fare']]
    # fit our model
    #model.fit(X, y)
    #pred = model.predict(y)

    # display metrices
    #display.subheader("Mean Absolute error of the model is:")
    #display.write(mean_absolute_error(y, pred))
    #display.subheader("Mean Squared error of the model is:")
    #display.write(mean_squared_error(y, pred))
    #display.subheader("r2 Score of the model is:")
    #display.write(r2_score(y, pred))
    # Validate input feature
    if input_features not in df.columns:
        st.error(f"'{input_features}' is not a valid feature. Please select from the available features.")
    else:
        # Define X and y
        X = df[[input_features]]  # Single feature selected by the user
        y = df[['fare']]
        
        # Train model
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
        model.fit(X, y)
        
        # Predict
        pred = model.predict(X)  # Use the same `X` for prediction
        
        # Display metrics
        
        display.subheader("Mean Absolute error of the model is:")
        display.write(mean_absolute_error(y, pred))
        display.subheader("Mean Squared error of the model is:")
        display.write(mean_squared_error(y, pred))
        display.subheader("r2 Score of the model is:")
        display.write(r2_score(y, pred))
    