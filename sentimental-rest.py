# Import the necessary libraries
from flask import Flask, request, jsonify
from textblob import TextBlob
from statistics import mean
# import snscrape.modules.twitter as sntwitter
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from flask_cors import CORS
nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app)
# query = "#TSLA "  # Hashtag to search for, filtered by English tweets


# limit = 5000
# count = 0
# sleep_time = 600  # 10 minutes
# tweets = []
# for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query + "lang:en").get_items()):
#     if i > limit:
#         break
#     else:
#         tweets.append(
#             [tweet.date, tweet.user.username, tweet.rawContent, ])
#         print('Tweet scraped:', count)


# # Create a dataframe from the tweets list above
# tweets_df = pd.DataFrame(tweets, columns=['Datetime', 'Username', 'Text'])

# # Export dataframe into a CSV
# tweets_df.to_csv('tweets.csv', sep=',', index=False, encoding='utf-8')

# Run sentiment analysis on the tweets to see if they are positive or negative
# Create a sentiment analyzer

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('tweets.csv')

# Define a function to get the sentiment of a tweet using TextBlob


df = pd.read_csv('tweets.csv')


def get_sentiment(tweet):
    blob = TextBlob(tweet)
    sentiment = blob.sentiment.polarity
    return sentiment


df['sentiment'] = df['Text'].apply(get_sentiment)

# Calculate the average sentiment score of the tweets
avg_sentiment = df['sentiment'].mean()

# Print the average sentiment score
print('Average sentiment score:', avg_sentiment)

# Classify the tweets as positive or negative based on the sentiment score
df['sentiment_label'] = df['sentiment'].apply(
    lambda x: 'positive' if x > 0 else 'negative')

# Save the updated DataFrame to a new CSV file
df.to_csv('sentiment_analysis.csv', index=False)


@app.route('/sentiment', methods=['GET'])
def sentiment():
    return jsonify({'sentiment': avg_sentiment})


if __name__ == '__main__':
    app.run()
