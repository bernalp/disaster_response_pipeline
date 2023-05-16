#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries

# In[1]:


# import libraries
import pandas as pd


# - Load `messages.csv` into a dataframe and inspect the first few lines.

# In[2]:


# load messages dataset
df_messages = pd.read_csv('messages.csv')
df_messages.head()


# - Load `categories.csv` into a dataframe and inspect the first few lines.

# In[3]:


# load categories dataset
df_categories = pd.read_csv('categories.csv')
df_categories.head()


# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps

# In[4]:


# merge datasets based on their 'id'
df = df_messages.merge(df_categories, how='outer',                            on=['id'])

df.head()


# In[5]:


#check dataframe types
df.dtypes


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.

# In[6]:


# create a dataframe of the 36 individual category columns
# assuming the categories are in a column named 'categories' of a DataFrame named 'df_categories'
category_series = df_categories['categories']
categories = category_series.str.split(";", expand=True)
categories.head()


# - Use the first row of categories dataframe to create column names for the categories data.

# In[7]:


# select the first row of the categories dataframe
first_row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = first_row.apply(lambda x: x[:-2]).tolist()
print(category_colnames)


# - Rename columns of `categories` with new column names.

# In[8]:


# rename the columns of `categories`
categories.columns = category_colnames
categories.head()


# ### 4. Convert category values to just numbers 0 or 1.
# - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
# - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

# In[9]:


for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])

categories.head()


# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.

# In[10]:


# drop the original categories column from `df`
df = df.drop('categories', axis=1)

df.head()


# - Concatenate df and categories data frames.

# In[11]:


# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)
df.head()


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.

# In[12]:


# check number of duplicates
print("Number of duplicates: ", df.duplicated().sum())


# - Drop the duplicates.

# In[13]:


# drop duplicates
df = df.drop_duplicates()


# - Confirm duplicates were removed.

# In[14]:


# check number of duplicates after dropping
print('Number of duplicates after dropping:', df.duplicated().sum())


# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.

# In[15]:


# import SQLAlchemy's create_engine
from sqlalchemy import create_engine

# create an engine to connect to the database
engine = create_engine('sqlite:///disaster_response.db')

# save the clean dataset to a new table in the database
df.to_sql('messages_categories', engine, index=False, if_exists='replace')

import pandas as pd
from sqlalchemy import create_engine

# Create an engine to connect to the database
engine = create_engine('sqlite:///disaster_response.db')

# Query the database and fetch the data from the 'messages_categories' table
query = "SELECT * FROM messages_categories"
df = pd.read_sql(query, engine)

# Save the DataFrame to a CSV file
df.to_csv('messages_categories.csv', index=False)
# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.

# In[16]:


import pandas as pd
from sqlalchemy import create_engine

# Get input file paths from the user
messages_filepath = input("Enter the path to the messages dataset: ")
categories_filepath = input("Enter the path to the categories dataset: ")
database_filepath = input("Enter the path for the output database file: ")

# Load the datasets
df_messages = pd.read_csv('messages.csv')
df_categories = pd.read_csv('categories.csv')

# Merge datasets based on 'id'
df = df_messages.merge(df_categories, how='outer', on=['id'])

# Create individual category columns
category_series = df['categories']
categories = category_series.str.split(";", expand=True)
category_colnames = categories.iloc[0].apply(lambda x: x[:-2]).tolist()
categories.columns = category_colnames

# Convert category values to numeric
for column in categories:
    categories[column] = categories[column].str[-1]
    categories[column] = pd.to_numeric(categories[column])

# Drop the original 'categories' column from df
df = df.drop('categories', axis=1)

# Concatenate the original dataframe with the new 'categories' dataframe
df = pd.concat([df, categories], axis=1)

# Drop duplicates
df = df.drop_duplicates()

# Create an engine to connect to the database
engine = create_engine('sqlite:///disaster_response.db')

# Save the clean dataset to a new table in the database
df.to_sql('messages_categories', engine, index=False, if_exists='replace')

# Confirm the process completion
print("Database created successfully!")


# In[ ]:




