#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
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

# setup regular expression
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

print('\n', 'Import libraries and setup regular expression complete')


# In[2]:


# load data from database
def load_data():
    engine = create_engine('sqlite:///disaster_response.db')
    df = pd.read_sql_table('messages_categories', engine)
    df.dropna(inplace=True)
    X = df['message']
    y = df.iloc[:, 4:]
    return X, y

load_data()


# ### 2. Write a tokenization function to process your text data

# In[3]:


# setup tokenization function
def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

X, y = load_data()
for message in X[:5]:
    tokens = tokenize(message)
    print(message)
    print(tokens, '\n')

print('Setup tokenization function complete')


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[4]:


import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputClassifier

# define pipeline steps
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

print('\n', 'Setup pipeline complete')


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[5]:


# load data
X, y = load_data()

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train pipeline
pipeline.fit(X_train, y_train)

# predict on test data
y_pred = pipeline.predict(X_test)

print('\n', 'Split data and train pipeline complete')


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[6]:


from sklearn.metrics import classification_report

# make predictions on test data
y_pred = pipeline.predict(X_test)

# iterate over columns and print classification report for each
for i, col in enumerate(y_test.columns):
    print(f"Category: {col}")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
    


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[7]:


from sklearn.model_selection import GridSearchCV

# define the parameter grid
parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'clf__estimator__n_estimators': [10, 20],
    'clf__estimator__min_samples_split': [2, 4],
}

# perform grid search with 5-fold cross validation
cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1)

# fit the grid search object to the training data
cv.fit(X_train, y_train)

# print the best parameters and score
print("Best Parameters:", cv.best_params_)
print("Best Score:", cv.best_score_)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[8]:


import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

def build_model():
    """
    Build and return a GridSearchCV object using a pipeline that preprocesses text data and fits a random forest 
    classifier to classify multiple target labels.
    
    Returns:
    cv (GridSearchCV): A GridSearchCV object with the specified pipeline and parameters.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 4],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1)
    return cv


def load_data():
    """
    Load the data from the SQLite database and split it into features (X) and targets (y).
    
    Returns:
    X (pandas DataFrame): The features (message column) of the loaded data.
    y (pandas DataFrame): The targets (all columns except message) of the loaded data.
    """
    engine = create_engine('sqlite:///disaster_response.db')
    df = pd.read_sql_table('messages_categories', engine)
    df.dropna(inplace=True)
    X = df['message']
    y = df.iloc[:, 4:]
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
    The main function that loads the data, builds and trains a model, and displays the accuracy and best parameters
    of the model using test data.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    display_results(model, y_test, y_pred)


if __name__ == '__main__':
    main()


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[9]:


import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

# Function to extract the length of text messages
def get_text_length(X):
    return np.array([len(text) for text in X]).reshape(-1, 1)

def build_model():
    """
    Build and return a GridSearchCV object using a pipeline that preprocesses text data and fits a random forest 
    classifier to classify multiple target labels. Includes an additional feature transformer for text length.

    Returns:
    cv (GridSearchCV): A GridSearchCV object with the specified pipeline and parameters.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('text_length', FunctionTransformer(get_text_length, validate=False)),
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 4],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1)
    return cv


def load_data():
    """
    Load the data from the SQLite database and split it into features (X) and targets (y).
    
    Returns:
    X (pandas DataFrame): The features (message column) of the loaded data.
    y (pandas DataFrame): The targets (all columns except message) of the loaded data.
    """
    engine = create_engine('sqlite:///disaster_response.db')
    df = pd.read_sql_table('messages_categories', engine)
    df.dropna(inplace=True)
    X = df['message']
    y = df.iloc[:, 4:]
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
    The main function that loads the data, builds and trains a model, and displays the accuracy and best parameters
    of the model using test data.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    display_results(model, y_test, y_pred)


if __name__ == '__main__':
    main()


# ### 9. Export your model as a pickle file

# In[10]:


import pickle

def main():
    """
    The main function that loads the data, builds and trains a model, and displays the accuracy and best parameters
    of the model using test data. It also exports the trained model as a pickle file.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    display_results(model, y_test, y_pred)

    # Save the trained model as a pickle file
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    main()


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[12]:


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


# In[ ]:




