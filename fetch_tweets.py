import requests, csv
import os
import json

bearer_token = os.getenv('MY_TOKEN')
fetch_tweets_url = "https://api.twitter.com/2/users/{0}/tweets"
username_url = "https://api.twitter.com/2/users/by/username/{0}"

query_params_get_tweets = {"max_results": 20}

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return json.loads(response.content)

def get_user_id_by_username(username):
    response = connect_to_endpoint(username_url.format(username), {})
    if response.get('data') is not None:
        return response.get('data').get('id')
    else:
        return None

def get_tweets_by_user_id(user_id):
    response = connect_to_endpoint(fetch_tweets_url.format(user_id), query_params_get_tweets)
    if response is not None:
        tweets = response.get('data')
        tweet_text_list = []
        for tweet in tweets:
            tweet_text_list.append(tweet.get('text'))
        return tweet_text_list
    else:
        return None

def make_tweets_csv(tweets, username):
    with open('tweets_{0}.csv'.format(username), 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tweet"])
        for tweet in tweets:
            lst = []
            lst.append(tweet)
            writer.writerow(lst)
    csvfile.close()

    