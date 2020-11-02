# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:53:45 2020

@author: heman

TEXT CLASSIFICATION

You did a great such a great job for DeFalco's restaurant in the previous exercise that 
the chef has hired you for a new project.

The restaurant's menu includes an email address where visitors can give feedback about their food.

The manager wants you to create a tool that automatically sends 
him all the negative reviews so he can fix them, while automatically
 sending all the positive reviews to the owner, so the manager can ask for a raise.

You will first build a model to distinguish positive reviews from
 negative reviews using Yelp reviews because these reviews include 
 a rating with each review. Your data consists of the text body of each review
 along with the star rating. Ratings with 1-2 stars count as "negative", and ratings
 with 4-5 stars are "positive". Ratings with 3 stars are "neutral" and have been dropped 
 from the data.

Let's get started. First, run the next code cell.
"""
#step 1 : review data and create the model
import pandas as pd

def load_data(csv_file,split =0.9):
    data = pd.read_csv(csv_file)
    
    #shuffle data
    train_data = data.sample(frac = 1,random_state=7)
    texts = train_data.text.values
    labels = [{"POSITIVE" : bool(y) ,"NEGATIVE" : not bool(y)}
              for y in train_data.sentiment.values]
    split = int(len(train_data) * split)
    
    train_labels = [{'cats':labels} for labels in labels[:split]]
    val_labels = [{'cats':labels} for labels in labels[split:]]
    return texts[:split],train_labels,texts[split:],val_labels
train_texts, train_labels, val_texts, val_labels = load_data('yelp_ratings.csv')
print(train_labels[:2])

import spacy

#load empty model
nlp =spacy.blank('en')

#make a text categorizer pipe

textcat = nlp.create_pipe('textcat',
                          config = {'exclusive_classes' : True,
                                    'architecture' : 'bow'})
#add the text categorizer to the empty model
    
nlp.add_pipe(textcat)

#add labels to text classifier
textcat.add_label('NEGATIVE')
textcat.add_label('POSITIVE')


#step 3 : train function

from spacy.util import minibatch
import random

def train(model,train_data,optimizer):
    losses = {}
    random.seed(1)
    random.shuffle(train_data)
    batches = minibatch(train_data,size=8)
    for batch in batches:
        #train data is in tuples [(text 0,label 0),(text 1,label 1-----)
        #split the data into text and labels
        texts,labels = zip(*batch)
         
        # Update model with texts and labels
        model.update(texts,labels,sgd = optimizer,losses=losses)
        
    return losses
# Fix seed for reproducibility
spacy.util.fix_random_seed(1)
random.seed(1)

# This may take a while to run!
optimizer = nlp.begin_training()
train_data = list(zip(train_texts, train_labels))
losses = train(nlp, train_data, optimizer)
print(losses['textcat'])

text = "This tea cup was full of holes. Do not recommend."
doc = nlp(text)
print(doc.cats)

def predict(nlp,texts):
    #use the models tokenizer to tokenize the input text
    docs = [nlp.tokenizer(text) for text in texts]
    #use textcat to get the scores foreach doc
    
    textcat = nlp.get_pipe('textcat')
    scores,_=textcat.predict(docs)
     # From the scores, find the class with the highest score/probability
    predicted_class = scores.argmax(axis=1)
    return predicted_class

texts =val_texts[34:38]
predictions = predict(nlp,texts)
for p,t in zip(predictions,texts):
    print(f"{textcat.labels[p]} : {t} \n")
 
    
def evaluate(model,texts,labels):
    """ Returns the accuracy of a TextCategorizer model. 
        
            Arguments
            ---------
            model: ScaPy model with a TextCategorizer
            texts: Text samples, from load_data function
            labels: True labels, from load_data function
        
        """
    # Get predictions from textcat model (using your predict method)    
    predicted_class = predict(model,texts)
     # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)
    true_class = [int(each['cats']['POSITIVE']) for each in labels]
    # A boolean or int array indicating correct predictions
    
    correct_predictions = predicted_class == true_class
    # The accuracy, number of correct predictions divided by all predictions
    accuracy = correct_predictions.mean()
    return accuracy

        
accuracy = evaluate(nlp, val_texts, val_labels)
print(f"Accuracy: {accuracy:.4f}")




n_iters = 5
for i in range(n_iters):
    losses = train(nlp, train_data, optimizer)
    accuracy = evaluate(nlp, val_texts, val_labels)
    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")
