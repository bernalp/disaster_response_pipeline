import sys


def load_data(messages.csv, categories.csv):
    pass


def clean_data(df):
    pass


def save_data(df, disaster_response.db):
    pass  


def main():
    if len(sys.argv) == 4:

        messages.csv, categories.csv, disaster_response.db = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages.csv, categories.csv))
        df = load_data(messages.csv, categories.csv)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(disaster_response.db))
        save_data(df, disaster_response.db)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()