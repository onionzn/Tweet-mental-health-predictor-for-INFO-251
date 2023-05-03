import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import TFBertModel
import tensorflow as tf
from tensorflow import keras
from transformers import TFDistilBertModel
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from pathlib import Path

THIS_FOLDER = Path(__file__).parent.resolve()

def make_prediction_for_username(username, distilbert=False):

    # Load the tweets dataset
    df_tweets = pd.read_csv('tweets_{0}.csv'.format(username))

    # Load training dataset
    if distilbert:
        train_df = pd.read_csv(THIS_FOLDER / 'train_df_distilbert.csv')
        test_df = pd.read_csv(THIS_FOLDER / 'test_df_distilbert.csv')
    else:
        train_df = pd.read_csv(THIS_FOLDER / 'train_df_bert.csv')
        test_df = pd.read_csv(THIS_FOLDER / 'test_df_bert.csv')
   
    # Tokenize the tweets
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tweet_encodings = tokenizer(df_tweets['tweet'].tolist(), truncation=True, padding=True)

    # Get features and attention mask from tweet_encodings
    tweet_features = np.array(tweet_encodings['input_ids'])
    tweet_attention_mask = np.array(tweet_encodings['attention_mask'])

    # Create embeddings for tweets
    if distilbert:
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    else:
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    tweet_embeddings = bert_model([tweet_features, tweet_attention_mask])[0][:, 0, :]

    # Load the saved embeddings from files
    if distilbert:
        train_embeddings = np.load(THIS_FOLDER / 'train_embeddings_distilbert_1500rows.npy')
        test_embeddings = np.load(THIS_FOLDER / 'test_embeddings_distilbert_1500rows.npy')
    else:
        train_embeddings = np.load(THIS_FOLDER / 'train_embeddings_bert_1500rows.npy')
        test_embeddings = np.load(THIS_FOLDER / 'test_embeddings_bert_1500rows.npy')

    # Define the neural network model
    if distilbert:
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(768,), 
                            kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.1),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.1),
            keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid')
        ])
    else:
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(768,), 
                            kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.3),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.2),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
            Dropout(0.2),
            keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
            Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    # Compile the neural network model
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                metrics=['accuracy'])
    
    # Train the neural network model
    if distilbert:
        model.fit(train_embeddings, train_df['label'], epochs=50, batch_size=32,
            validation_data=(test_embeddings, test_df['label']))
    else:
        model.fit(train_embeddings, train_df['label'], epochs=100, batch_size=32,
            validation_data=(test_embeddings, test_df['label']))
    
    # Make predictions on tweets
    predictions_prob = model.predict(tweet_embeddings)
    threshold = 0.5
    predictions_class = (predictions_prob >= threshold).astype(int)

    return predictions_class