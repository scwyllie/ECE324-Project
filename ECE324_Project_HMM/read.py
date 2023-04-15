
# https://github.com/weizhaoalex/EnglishAlphabetMarkovModel/blob/master/English%20Alphabet%20Markov%20Model.ipynb
import numpy as np
from collections import Counter 
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns

def read_book(title_path):
    """
    Read a book.
    """
    with open(title_path,"r",encoding="utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n","").replace("\r","")
    return text


book = read_book("./tale_of_two_cities.txt")
# Choose a book to save into a string variable for further analysis, and convert all characters into lower case. 
# You can concatenate multiple books together to get a larger data set.
text = book.lower()

# create a list contains the major punctuations that we want to skip
skips = [".",":",";","'",",",'"',"!","?","”","“","_","(",")","-","’"]

for ch in skips:
    text = text.replace(ch," ")

# # Create a dataframe of the word frequency in that book, and sort it    

word_count = Counter(text.split(" "))

text_stat = pd.DataFrame({"word":list(word_count.keys()),
                           "count":list(word_count.values())})

text_stat = text_stat[text_stat.word != ""]

text_stat = text_stat.sort_values(by=['count'], ascending = False)

# Show the top frequency words
text_stat.head()
alp = string.ascii_lowercase

# Initialize a 26*26 transition matrix
s=(27,27)
matrix = pd.DataFrame(np.zeros(s))

# Initialize a 26*1 initial letter vector
t=(27)
Initial = pd.DataFrame(np.zeros(t))

# Loop over the word frequency data frame to calculate the initial vector and transition matrix
# It may take a few sec depends on the size of the dataframe
# we will use alp to convert each character into an index: a - 0, b - 1, ..., z - 25
for i in range(len(text_stat.iloc[:,0])):
    word = text_stat.iloc[i,0]
    if len(str(word))>1:
        pos_i = alp.find(word[0])
        Initial.iloc[pos_i] += text_stat.iloc[i,1]
    for j in range(len(word)-1):
        pos1 = alp.find(word[j])
        pos2 = alp.find(word[j+1])
        matrix.iloc[pos1,pos2]+=text_stat.iloc[i,1]
        
np.savetxt("bigram.csv", matrix, delimiter=",")
Initial_dist = Initial.div(Initial.sum(axis=0),axis=1)
Initial_dist.columns = ['Prob']

Alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','space']
Initial_dist['Alphabet'] = Alphabet

Initial_dist.plot.bar(x='Alphabet', y='Prob')
plt.figure(figsize=(10,6))
barchart = sns.barplot(x="Alphabet", y="Prob", data=Initial_dist)
plt.show()