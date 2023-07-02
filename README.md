
# Disaster Response Pipeline Project

## Table of Contents

- **1.** [Installation](#installation)
- **2.** [Project Motivation](#motivation)
- **3.** [Instructions](#instructions)
- **4.** [Acknowledgments](#acknow)

### 1. Installation <a name="installation"></a>
To run the code successfully, make sure we have these libraries installed in our Python environment:

    - pandas
    - sqlalchemy
    - re
    - nltk
    - string
    - numpy
    - joblib
    - sklearn

### 2. Project Motivation <a name="motivation"></a>
This is the second project from [Udacity](https://www.udacity.com/) "Data Science Nanodegree Program". I basically building an interface where you can enter a disaster message, and the system will automatically figure out what type of message it is. This will help in provide the right response and actions during times of disaster. This project is part of the Data Science Nanodegree Program by Udacity, in collaboration with Figure Eight. Dataset are combination tweets and messages from real-life disaster situations.

By using machine learning techniques, we can analyze and categorize the incoming messages quite fast. This means we can respond to disasters more efficiently and quickly. I'll guide you through the whole process of setting up the project. It's cover things like load the dataset, train the machine learning model, and run the web interface. Let's get started!

### 3. Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
         `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

<span>
<img src="https://github.com/bernalp/disaster_response_pipeline/blob/main/sc1.png" width=400px height="280px" />
<img src="https://github.com/bernalp/disaster_response_pipeline/blob/main/sc2.png" width=400px height="280px" />
</span>


### 4. Acknowledgements <a name="acknow"></a>
- Thanks to [Appen (formally Figure 8)](https://appen.com/press-release/appen-completes-acquisition-of-figure-eight-and-achieves-critical-integration-milestone/) for providing the dataset.
- Also [Udacity](https://www.udacity.com/) mentors who very helpful with their advice during project work.


