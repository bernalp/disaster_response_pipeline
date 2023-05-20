import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import joblib
from joblib import dump, load

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    """
    Tokenize and lemmatize the given text.

    Args:
    text (str): Input text to be tokenized and lemmatized.

    Returns:
    list: List of cleaned tokens.

    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def load_data():
    """
    Load data from the database.

    Returns:
    tuple: Dataframe, X, and y.

    """
    engine = create_engine('sqlite:///disaster_response.db')
    df = pd.read_sql_table('messages_categories', engine)
    df.dropna(inplace=True)
    X = df['message']
    y = df.iloc[:, 4:]
    return df, X, y


# load model
model = joblib.load("../models/trained_model.pkl")


@app.route('/')
@app.route('/index')
def index():
    """
    Display cool visuals and receive user input text for model.

    Returns:
    render_template: Rendered HTML template.

    """
    _, X, y = load_data()
    genre_counts = y.sum().values
    genre_names = y.columns.values

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """
    Handle user query and display model results.

    Returns:
    render_template: Rendered HTML template.

    """
    query = request.args.get('query', '')
    _, X, _ = load_data()
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(model.classes_, classification_labels))
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Main function to run the Flask application.

    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
    
    
