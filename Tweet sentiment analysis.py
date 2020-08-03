#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:10:58 2020

@author: valdaskniuipys
"""

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize 
from nltk import FreqDist, classify, NaiveBayesClassifier 

import re, string, random
import pandas as pd
import csv


#Removing the noise. 
# -> this step takes a word and makes it like a special token which has a tag. 
# -> it is required to remove the special characters like links and other symbols.
# -> lemmatizer analyses the basis of the word (using the tag), to convert it to the normal form

def remove_noise(tweet_tokens, stop_words = ()):
    
    cleaned_tokens = []
   
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+","", token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        
        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos= "a"
            
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        
        if len(token)>0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# -> this is the function to get the words on all tweets where we can generate the frequency table
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# -> this function converts the tweets from cleaned list to dictionaries where word is a key and True is the value
# -> it is just what the Naive Bayes classifier requires. 
# -> Naive Bayes classifier is the probabilistic classifier. It is trained given the set of training data
            
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
 
#Training the model part.
# -> this includes importing the tweets that are already categorized and inserting it in the functions above.
if __name__ == "__main__" :
    
    positive_tweets = twitter_samples.strings("positive_tweets.json")
    negative_tweets = twitter_samples.strings("negative_tweets.json")
    text = twitter_samples.strings("tweets.20150430-223406.json")
    tweet_tokens = twitter_samples.tokenized("positive_tweets.json")[0]
    
    stop_words = stopwords.words("english")
    
    
    positive_tweet_tokens = twitter_samples.tokenized("positive_tweets.json")
    negative_tweet_tokens = twitter_samples.tokenized("negative_tweets.json")
    
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

# -> here is where you use the remove_noise function and append it to the cleaned list.   
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
        
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

# -> here you just use the get_all_words function indicated above to get the frequency distribution
# -> this frequency distribution is from the training datatsed. It is not tailored specifically to the Trump
# -> hence, I'll further include the set of specifying characteristics. 
        
    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    all_neg_words = get_all_words(negative_cleaned_tokens_list)
    
    freq_dist_pos = FreqDist(all_pos_words)
    freq_dist_neg = FreqDist(all_neg_words)
    print(freq_dist_pos.most_common(10))
    print(freq_dist_neg.most_common(10))

# -> this uses the dictionary conversion function. 
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    
    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]
    
    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]
    
    dataset = positive_dataset + negative_dataset
    
    random.shuffle(dataset)
    
    train_data = dataset[:8000]
    test_data = dataset[8000:]
 
# -> this uses the training datased our the given sample tweets. It learns what is positive or negative. 
# -> then it tests it on the training dataset and we can see the accuracy. 
# -> this accuracy is less of the use for me. 
    
    classifier = NaiveBayesClassifier.train(train_data)
    
    print("Accurcy is: ", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))

# -> importing twitter and firm data 
    df = pd.read_excel(r'/Users/valdaskniuipys/Desktop/Cleaned twitter sample.xlsx')
    tweet_list = df["text"].values.tolist()
    
    df = pd.read_excel(r'/Users/valdaskniuipys/Desktop/S&P firms.xlsx')
    Firms_list = df["S&P 500 firms"].values.tolist()
    
# -> lists for sorting    
    Economy = ["tariff", "tariffs", "retaliate", "tax", "jobs", "booming", "boom", "interest", "inflation", "trade balance", "supply", "demand", "downturn", "bubble", "budget", "fiscal", "GDP", "gdp", "recession", "inflation", "quantitative venice", "sector", "wages", "wage", "development", "trade", "vat", "currency", "china", "deal", "mexico", "fed", "oil", "economy"]
    Stock_market = ["stock", "market", "markets", "nasdaq", "S&P", "dow", "DOW", "trading", "wallstreet", "%", "investor", "asset", "management", "bearish", "bullish", "valuation", "equity", "equities", "bond", "bonds", "dividend", "shares", "finance",  "libor", "margin", "fund", "sec", "revenue", "offshore", "vix", "volatility", "merger", "venture", "capital", "yield", "dividends", "portfolio"]
    Politics = ["democrats", "republicans", "election", "covfefe", "collusion", "report", "witch", "hunt", "senator", "border", "fake", "media", "country", "military", "russia", "korea", "kim", "nation", "agreement", "vote", "flag", "regime", "nuclear", "power", "countries", "president", "elections", "win", "poll", "weak", "fbi", "me", "myself", "great", "mexicans", "hackers", "hillary", "guns", "shooting", "support", "impeachment", "impeach", "americans", "ivanka", "association", "union", "leader", "leadership", "innocent", "Syria", "Turkey", "europe", "loser", "approval", "cia", "news", "obama", "migrant", "illegal", "xi", "war", "constitution", "congress", "pelosi", "caucus", "candidate", "thank", "amendment", "endorsement", "senators", "campaign", "delegates", "rally", "voter"]
    other_firms = ["ford", "chrysler", "toyota", "motors", "harley", "tesla", "crowdstrike", "wells", "fargo", "airbus", "inditex", "zara", "samsung", "volkswagen", "crowdstrike", "bean", "rexnord", "boeing", "fiat", "macys", "koch", "shell", "hsbc", "allianz", "bp", "santander", "daimler", "nestle", "novartis", "siemens", "enel"]
    industries = ["car", "airlines", "steel", "food", "farmers", "telecommunications", "electronics", "fossil", "medicine", "soybeans", "technology", "insurance"]
    Firms = Firms_list + other_firms

# -> function to further categorize the tweet: 
    def classification(custom_token):
        classi = []
        for token in custom_token:
            if token in Economy and "Economy" not in classi:
                classi.append("Economy")
            if token in Stock_market and "Stock market" not in classi:
                classi.append("Stock market")
            if token in Politics and "Politics" not in classi:
                classi.append("Politics")
            if token in Firms and "Firms" not in classi:
                classi.append("Firms")
            if token in industries and "Industry" not in classi:
                classi.append("Industry")
        return str(classi)

# this is used to open the text file and write the output there. 
    with open("/Users/valdaskniuipys/Desktop/Book3.txt", "w") as file:
        for custom_tweet in tweet_list:
            custom_tokens = remove_noise(word_tokenize(custom_tweet))
            output = [custom_tweet, "*", classifier.classify(dict([token, True] for token in custom_tokens)), classification(custom_tokens), "\n"]
            file.writelines(output)
    file.close()

