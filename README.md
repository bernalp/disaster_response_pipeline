
# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data / messages.csv data / categories.csv data / disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models / train_classifier.py data / disaster_response.db models / classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
