import pandas as pd
import numpy as np 
import pickle
import yaml
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load


def load_data(train_data, test_data):
    # Load data
    train_data.to_csv("data/train.csv", index=False)
    test_data.to_csv("data/test.csv", index=False)
    return train_data, test_data

def process_data(train_df, test_df):
    train_df['text'] = train_df['title'] + " " + train_df['content']
    test_df['text'] = test_df['title'] + " " + test_df['content']

    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_df['text'], 
        train_df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['label']
    )

    tfidf_vectorizer = TfidfVectorizer(max_features=500, min_df=5, max_df=0.7)
    X_train = tfidf_vectorizer.fit_transform(train_data)
    X_test = tfidf_vectorizer.transform(test_df['text'])

    


    return  X_train, X_test, tfidf_vectorizer

def save_data(train_new, test_new, train_name, test_name, tfidf_vectorizer):
    X_train_df = pd.DataFrame(train_new.toarray())
    X_train_df.to_csv(train_name, index=False)
    X_test_df = pd.DataFrame(test_new.toarray())
    X_test_df.to_csv(test_name, index=False)
    
    # Save pipeline
    dump(tfidf_vectorizer, 'data/tfidf_vectorizer.joblib')
    pickle.dump(train_new, open('data/train_data.pkl', 'wb'))
    pickle.dump(test_new, open('data/test_data.pkl', 'wb'))


if __name__=="__main__":
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    train_df, test_df, tfidf_vectorizer = process_data(train_df, test_df)
    save_data(train_df, test_df, "data/train_new.csv", "data/test_new.csv", tfidf_vectorizer)