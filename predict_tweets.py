import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import TFBertModel
import tensorflow as tf
from tensorflow import keras
from transformers import TFDistilBertModel
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

def make_prediction_for_username_distilbert_nn(username):
    # Load the dataset
    df = pd.read_csv('mental_health.csv')
    df = df.iloc[:1500]

    # Load the tweets dataset
    df_tweets = pd.read_csv('tweets_{0}.csv'.format(username))

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
   
    # Tokenize the text data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tweet_encodings = tokenizer(df_tweets['tweet'].tolist(), truncation=True, padding=True)

    # Get features and attention mask
    tweet_features = np.array(tweet_encodings['input_ids'])
    tweet_attention_mask = np.array(tweet_encodings['attention_mask'])

    # Create embeddings
    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')    
    tweet_embeddings = bert_model([tweet_features, tweet_attention_mask])[0][:, 0, :]

    # Load the saved embeddings from files
    train_embeddings = np.load('train_embeddings_distill_1500rows.npy')

    # Define the neural network model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(768,), 
                       kernel_regularizer=regularizers.l2(0.0001)),
        Dropout(0.1),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        Dropout(0.1),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.1),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the neural network model
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                metrics=['accuracy'])

    # Train the neural network model
    model.fit(train_embeddings, train_df['label'], epochs=100, batch_size=32)

    # Make predictions on tweets
    predictions_prob = model.predict(tweet_embeddings)
    threshold = 0.5
    predictions_class = (predictions_prob >= threshold).astype(int)

    return predictions_class