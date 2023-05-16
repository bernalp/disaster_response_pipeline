import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories datasets from the specified file paths.

    Args:
        messages_filepath (str): File path for the messages dataset.
        categories_filepath (str): File path for the categories dataset.

    Returns:
        pandas.DataFrame: Merged dataframe of the messages and categories datasets.
    """
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    df = df_messages.merge(df_categories, how='outer', on=['id'])
    return df


def clean_data(df):
    """
    Clean the merged dataframe by creating individual category columns, converting category values to numeric,
    dropping the original 'categories' column, concatenating the original dataframe with the new 'categories'
    dataframe, and dropping duplicates.

    Args:
        df (pandas.DataFrame): Merged dataframe of the messages and categories datasets.

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    category_series = df['categories']
    categories = category_series.str.split(";", expand=True)
    category_colnames = categories.iloc[0].apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df



def save_data(df, database_filepath):
    """
    Save the cleaned dataframe to a new table in the specified SQLite database file.

    Args:
        df (pandas.DataFrame): Cleaned dataframe.
        database_filepath (str): File path for the output SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')
    print("Database created successfully!") 


def main():
    """
    Main function that executes the data processing pipeline.
    Prompts the user to enter file paths for the messages dataset, categories dataset, and the desired
    output database file. Calls the load_data, clean_data, and save_data functions in the correct sequence.
    """
    messages_filepath = input("Enter the path to the messages dataset: ")
    categories_filepath = input("Enter the path to the categories dataset: ")
    database_filepath = input("Enter the path for the output database file: ")

    df = load_data(messages_filepath, categories_filepath)
    df = clean_data(df)
    save_data(df, database_filepath)

if __name__ == '__main__':
    main()