
# Disaster Response Pipeline Project

### Introduction:
The project basically building an interface where you can enter a disaster message, and the system will automatically figure out what type of message it is. This will help in provide the right response and actions during times of disaster. This project is part of the Data Science Nanodegree Program by Udacity, in collaboration with Figure Eight. Dataset are combination tweets and messages from real-life disaster situations.

By using machine learning techniques, we can analyze and categorize the incoming messages quite fast. This means we can respond to disasters more efficiently and quickly. I'll guide you through the whole process of setting up the project. It's cover things like load the dataset, train the machine learning model, and run the web interface. Let's get started!

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
         `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
