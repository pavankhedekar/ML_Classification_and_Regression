# Machine Learning Multi-Model Application

This project demonstrates a Streamlit application that can perform both classification and regression tasks using pre-trained machine learning models.

## Project Structure

The notebook `ACV4V6FzPu5-` and `HEzwYIYuSOEm` handles the data loading, preprocessing, and splitting for both classification (Iris dataset) and regression (Diabetes dataset). It also trains various models and identifies the best performing ones. The notebook `KN7yDDHn0RUg` and `2ptGpr9E4bfO` trains and evaluates multiple classification and regression models respectively. The notebook `T2a3Xs815Rx3` saves the best performing classification and regression models, along with their respective scalers, to `.pkl` files.

The Streamlit application (`5T5GHwsI7cyM`) loads these saved models and scalers to provide an interactive interface for users to make predictions.

## Features

- **Classification**: Predicts the class of Iris flowers based on sepal length, sepal width, petal length, and petal width.
- **Regression**: Predicts a target value for the Diabetes dataset based on 10 features.

## Models Used

### Classification (Iris Dataset)

- Logistic Regression (Selected as best model in `T2a3Xs815Rx3`)
- Decision Tree Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

### Regression (Diabetes Dataset)

- Linear Regression (Selected as best model in `T2a3Xs815Rx3`)
- Decision Tree Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor (KNN Regressor)

## How to Use

1.  **Run the Colab Notebooks**: Execute the cells in the notebooks to ensure all data preprocessing, model training, and model saving steps are completed.
2.  **Run the Streamlit Application**: The Streamlit app cannot be directly run within a Colab cell output. To run the Streamlit application, you would typically save the code from cell `5T5GHwsI7cyM` into a Python file (e.g., `app.py`) and then run it from your terminal using `streamlit run app.py`. In a Colab environment, you would need to set up a public URL to access the Streamlit app.
