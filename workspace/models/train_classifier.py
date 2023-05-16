import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split

from sqlalchemy import create_engine


def load_data(file_path):
    """
    Load the data from the specified file and return the features (X) and targets (y).

    Args:
    file_path (str): The path to the new dataset file.

    Returns:
    X (pandas DataFrame): The features (message column) of the loaded data.
    y (pandas DataFrame): The targets (all columns except message) of the loaded data.
    """
    # Load the data from the file into a pandas DataFrame
    df = pd.read_csv(file_path = 'messages_categories.csv')  # Modify this line based on the file format (e.g., pd.read_excel, pd.read_json)

    # Split the data into features (X) and targets (y)
    X = df['message']
    y = df.iloc[:, 4:]  # Modify this line based on the column indices or names of your target columns

    return X, y


def display_results(cv, y_test, y_pred):
    """
    Display the accuracy and best parameters of a trained model using the provided test targets and predicted targets.

    Args:
    cv (GridSearchCV): A trained GridSearchCV model.
    y_test (pandas DataFrame): The test targets.
    y_pred (numpy ndarray): The predicted targets.
    """
    labels = np.unique(y_pred)
    accuracy = (y_pred == y_test).mean()
    print("Labels:", labels)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)
    
    
def main():
    """
    The main function that prompts the user for a dataset, creates a database, loads the data into the database,
    builds and trains a model, and displays the accuracy and best parameters of the model using test data.
    It also exports the trained model as a pickle file.
    """
    # Prompt the user for the dataset file path
    dataset_path = input("Enter the path to the new dataset file: ")

    # Prompt the user for the desired database name
    database_name = input("Enter the desired name for the database file: ")

    # Create the database engine
    engine = create_engine('sqlite:///disaster_response.db')
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Build the model
    model = build_model()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Save the trained model as a pickle file
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Call the main function
if __name__ == '__main__':
    main()