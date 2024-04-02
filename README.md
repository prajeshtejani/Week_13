# Hate speech detection

Twitter Sentiment Analysis is a Python script designed to perform sentiment analysis on tweets using a logistic regression model. The script preprocesses the tweet data, trains the model, and predicts the sentiment of test data.
# Dependencies
- Python 3.6+
- pandas
- numpy
- matplotlib
- nltk
- scikit-learn
- wordcloud

You can install the dependencies using pip:
-      pip install pandas numpy matplotlib nltk scikit-learn wordcloud
Additionally, you need to download NLTK resources:
-      python -m nltk.downloader stopwords punkt

# Usage
- Ensure that you have the required dependencies installed.
- Place your training data in a CSV file named train_data.csv with columns 'tweet' and 'label'.
- Place your test data in a CSV file named test_data.csv with a column 'tweet'.
- Run the script twitter_sentiment_analysis.py.

# Description
The script performs the following steps:

- Reads the training and test data from CSV files.
- Preprocesses the tweet data by removing special characters, URLs, and stopwords, and applying stemming.
- Up-samples the minority class to handle class imbalance.
- Constructs word clouds to visualize the most common words in the training and test data.
- Builds a logistic regression model using a pipeline with CountVectorizer and TF-IDF transformer.
- Evaluates the model's performance using F1 score.
- Predicts the sentiment of the test data using the trained model.

# Results
The script achieves an F1 score of approximately 0.967 on the test data, indicating good performance in sentiment classification.
