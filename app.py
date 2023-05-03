from flask import Flask, render_template, request
from fetch_tweets import get_user_id_by_username, get_tweets_by_user_id, make_tweets_csv
from predict_tweets import make_prediction_for_username
import os
from builtins import zip

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        username = request.form['username']
        model = request.form['model']

        # Get user id
        user_id = get_user_id_by_username(username)
        if user_id is None:
            return render_template('homepage.html', user_not_found=True)
        
        # Fetch tweets
        tweets = get_tweets_by_user_id(user_id)
        if tweets is None:
            return render_template('homepage.html', no_tweets_available=True)
        
        # Save tweets into a .csv file
        make_tweets_csv(tweets, username)

        # Make predictions
        if model == 'distilbert_neural_networks':
            print("distil bert is used!!")
            predictions = make_prediction_for_username(username, True)
        else:
            predictions = make_prediction_for_username(username, False)

        # Delete the .csv file
        os.remove('./tweets_{0}.csv'.format(username))
        return render_template('homepage.html', tweets=tweets, predictions=predictions)
    else:
        # Render homepage
        return render_template('homepage.html')
    
if __name__ == '__main__':
    app.run()