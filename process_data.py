import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages, categories


def clean_data(messages, categories):
    
    #create clean files for categories
    
    #split categories data frame into 36 columns
    categories_clean = categories['categories'].str.split(";", expand=True)
    
    #add columns names
    row = categories_clean.iloc[0]
    category_colnames = list(map(lambda x: x[:-2],row))
    categories_clean.columns = category_colnames
    
    #replace information with jsu 0 and 1
    for column in categories_clean:
        categories_clean[column] = categories_clean[column].astype(str).str[-1:]
        categories_clean[column] = categories_clean[column].astype(int)

    #add IDs to clean categories df    
    categories_clean['id']=categories['id']    
   
    # merge the messages dataframe with the clean `categories` dataframe
    df = messages.merge(categories_clean, left_on='id', right_on='id')
    
    # drop duplicates
    df = df.drop_duplicates()
   
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql(database_filename, engine, if_exists='replace', index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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
