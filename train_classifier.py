import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

# download necessary NLTK data
import nltk

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# import statements

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
  
    df = pd.read_sql_table("DisasterResponse.db", engine)
    
    df = df.drop(columns=['child_alone'])
    X = df.message.values
    y = np.asarray(df[df.columns[4:]])
    return df, X, y


def tokenize(text):
    # normalize case and remove punctuation
    new_text = [word.lower() for word in text]
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text))
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens


def build_model():
    knn = KNeighborsClassifier(n_neighbors=3)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(knn, n_jobs=-1)),
    ])
    
    df, X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # train classifier
    pipeline.fit(X_train,y_train)

    # predict on test data
    y_pred = pipeline.predict(X_test)
    
    return y_test, y_pred


def evaluate_model(y_test, y_pred, category_names):
    print(classification_report(np.hstack(y_test_o),np.hstack(y_pred_o)))
    
    x = range(35)
    for n in x:
        y_test_n = y_test[...,n]
        y_pred_n = y_pred[...,n]
        print(classification_report(np.hstack(y_test_n),np.hstack(y_pred_n)))


def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
