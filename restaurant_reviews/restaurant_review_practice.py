# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:34:19 2020

@author: heman 

BASIC TEXT PROCESSING WITH SPACY

You're a consultant for DelFalco's Italian Restaurant.
 The owner asked you to identify whether there are any 
 foods on their menu that diners find disappointing.
"""

import pandas as pd

#read the data file
data = pd.read_json('restaurant.json')

data.head(5)


menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",
        "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",
        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",
        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",
        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",
        "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",
        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",
        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  
         "Prosciutto", "Salami"]

#step1 : plan your analysis

'''You could group reviews by what menu items they mention, and then calculate the average rating for reviews that mentioned each item. You can tell which foods are mentioned in reviews 
with low scores, so the restaurant can fix the recipe or remove those foods from the menu.'''

#step 2: Find items in one review

import spacy
from spacy.matcher import PhraseMatcher

index_of_review_to_test_on = 14
review = data.text.iloc[index_of_review_to_test_on]
nlp = spacy.load('en')

#tokenized version of single review
review_doc  = nlp(review)

#create object matcher of phrase matcher use attr 'LOWER'

matcher = PhraseMatcher(nlp.vocab,attr ='LOWER')

#create a list of tokens for each item in menu

menu_token_list = [nlp(item) for item in menu]

#add menu token list in matcher
matcher.add('MENU',None , *menu_token_list)

matches = matcher(review_doc)

for match in matches:
    print(f"Token number {match[1]} : {review_doc[match[1]:match[2]]}")


#step 3: matching the whole dataset
    
from collections import defaultdict

#item ratings is a dictionary of lists. if a key doesn't exist in item ratings 
# the key is added with an empty list as value
item_ratings = defaultdict(list)

#go through each review in the data and add rating of each matched item in a dictionary

for idx , review in data.iterrows():
    doc = nlp(review.text)
    
    #match patterns 
    matches =matcher(doc)
    #create a list of items which are matched
    found_items = set([doc[match[1]:match[2]].lower_ for match in matches])
    
    #append rating for each item in dictionary
    for item in found_items:
        item_ratings[item].append(review.stars)

print(item_ratings)

# Calculate the mean ratings for each menu item as a dictionary
mean_ratings = {item: sum(ratings)/len(ratings) for item, ratings in item_ratings.items()}

worst_item = sorted(mean_ratings,key=mean_ratings.get)[0]

print(worst_item)