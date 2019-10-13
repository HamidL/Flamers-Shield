#HACKUPC -2019 Flamers-Shield

## Inspiration
A large proportion of comments posted on the internet tend to be positive and can be described as constructive, but there is a small proportion of them could be considered toxic. And this is a severe problem because being on the internet could help anonymity, and that helps people be even more toxic than usual. 

As people who enjoy videogames, this toxic behaviour is very disruptive and may prevent you from enjoying your game.

## What it does
This project predicts the probability of a comment being one of the following labels:
* Toxic
* Severe toxic
* Obscene
* Threat
* Insult
* Identity hate

## How we built it
The main steps can be simplified in:
* Text preprocessing: Several preprocessing techniques have been applied and its output will be used to train the model. In order to do so, Python was used as the main framework.
* Modelling: A GLM and a 2RNN+CNN have been implemented. In this case, R was used.

## Challenges we ran into
First of all, we needed to find a dataset that contains many labelled examples. In this case, the Kaggle dataset [link](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) has been used. Also, the text preprocessing techniques have been quite difficult as we were not familiar with this.

Finally, we had to select the models. We wanted something that we were not really familiar with, and for this reason, we chose the aforementioned ones.

## Accomplishments that I'm proud of
Obtaining a final good model that is capable of classifying the comments and achieving an AUC of 
0.984 in the Kaggle ranking.

## What I learned
From the preprocessing:
* Different techniques of preprocessing (Stemming and Lemmatization)
* Tokenization methods (N-Gram, N-Hash-Gram, TFIDF, etc.)

From the modelling:
* How a GLM and a NN work.
* Cross-validation techniques
* Performance metrics (AUC, ROC, Accuracy)

## What's next for Flamers'Shield
* Apply this model in a real-time scenario.
* Explore other models and compare its performances.
