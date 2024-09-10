# Almond Types Classification App

This Streamlit application is designed to classify different types of almonds using a machine learning model. The app allows users to explore the dataset, adjust model hyperparameters, upload images, and visualize the model's performance.

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Demo

You can check out the deployed application on Streamlit Cloud: [Almond Types Classification App](https://your-streamlit-cloud-url)

## Features

- **Dataset Exploration**: View and explore the almond dataset used for classification.
- **Model Training**: Train a Random Forest classifier with adjustable hyperparameters.
- **Interactive Widgets**: Use sliders and input boxes to set model parameters.
- **Confusion Matrix Visualization**: Visualize model performance using a confusion matrix heatmap.
- **Media Upload**: Upload and display almond images.
- **User-Friendly Interface**: The application is designed to be intuitive and easy to use.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/almond-classification-app.git
    cd almond-classification-app
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    conda create --name almond-classification python=3.8
    conda activate almond-classification
    ```

3. Install dependencies
      bash
    pip install -r requirements.txt
    ```

4. **Run the application**:
    ```bash
    streamlit run app.py
    ```

5. Open your web browser and navigate to `http://localhost:8501` to view the app.

## Usage

### Model Training

- Adjust the number of estimators and maximum depth using the sliders in the sidebar.
- Click the "Train Model" button to start training the Random Forest model.
- View the accuracy score and confusion matrix once training is complete.

### Media Upload

- Use the file uploader to upload an image of an almond.
- The image will be displayed in the app for further analysis.

