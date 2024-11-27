# import libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# to give heading to the app
st.write("""
# Explore Different ML Model to find which is best?
""")

# to add sidebar for data set
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

# to add another sidebar to select the classifier
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)

# Now to make a function to upload dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y
#Now call the function or make it equal to X & y variables
X, y = get_dataset(dataset_name)
# to print the shape of data on webapp
st.write('Shape of Dataset: ', X.shape)
st.write('Number of Classes: ', len(np.unique(y)))

# Next function for classifier
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params
# Call the second function
params = add_parameter_ui(classifier_name)

# Another function to get classifier
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                        max_depth=params['max_depth'], random_state=1234)
    return clf

if st.checkbox('Show Code'):
    with st.echo():
        # Call the third function
        clf = get_classifier(classifier_name, params)

        # Now split the dataset into train data and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=1234)

        #Now fit the classifier to the training data
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Now to check the accuracy of model apply accuracy score matrics
        acc = accuracy_score(y_test, y_pred)


# Call the third function
clf = get_classifier(classifier_name, params)

# Now split the dataset into train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=1234)

#Now fit the classifier to the training data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Now to check the accuracy of model apply accuracy score matrics
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc:.2f}')

# Now add a plot 2 dimensional add slicer also using PCA
pca = PCA(2)
X_projected = pca.fit_transform(X)
# Now show data in 0 or 1 dimenssion with slicer
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)
          