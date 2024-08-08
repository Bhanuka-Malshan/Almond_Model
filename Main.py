import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the application
st.title("Almond Types Classification App")

# Function to load the dataset
@st.cache
def load_data():
    data = pd.read_csv("Almond.csv")  # Replace with the path to your dataset
    return data

# Load data
data = load_data()

# Display the dataset with an expander
with st.expander("Dataset Overview"):
    st.write(data.head())

# Sidebar for user inputs
st.sidebar.header("Model Hyperparameters")

# Input widgets for model parameters
n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100, help="The number of trees in the forest.")
max_depth = st.sidebar.slider("Maximum Depth", 1, 20, 10, help="The maximum depth of the tree.")
random_state = st.sidebar.number_input("Random State", value=42, help="Random seed for reproducibility.")

# Preprocessing
st.subheader("Preprocessing the Data")
X = data.drop('Type', axis=1)
y = data['Type']

# Display feature and target summaries
with st.expander("Feature Summary"):
    st.write(X.describe())

with st.expander("Target Summary"):
    st.write(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Train the model
st.subheader("Training the Model")
train_button = st.button("Train Model")

if train_button:
    with st.spinner('Training in progress...'):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success(f"Model Trained! Accuracy: **{accuracy:.2f}**")

        # Display Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

# Media file uploader (Image Upload)
st.subheader("Upload an Image of an Almond")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

