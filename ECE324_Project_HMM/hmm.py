### A demonstration on the HMM Model 

# pip install hmmlearn
import numpy as np
from hmmlearn import hmm

### "hmmlearn" is the library we are going to use. It will do most of the work. 

### Our input data here should be in the form of K x 1:
### [ P_1
###   P_2
###   P_3
###   ...
###   P_k    
### ] where P_i corresponds to the probability of that keystroke is in ith class.
### Assuming we have a sequence of N keystrokes, that will result in a K x N matrix
### Here is the initialization of the data:

K = 50  # Assume we have 50 clusters
N = 100 # Assume we have 100 keystrokes
H = 27  # We are considering 26 alphabets and space
### Initialization of the data_matrix
data_matrix = np.random.rand(K, 1) 
data_matrix /= np.sum(data_matrix)
for i in range(1, N):
    prob_vector = np.random.rand(K, 1)  
    prob_vector /= np.sum(prob_vector)  
    data_matrix = np.hstack((data_matrix, prob_vector))
    
### In the paper it said GaussianHMM will not perform well due to overfitting
### I guess we can test first then try to use dif models    
model = hmm.GaussianHMM(n_components= H, covariance_type="diag", n_iter=1000)

###
# In the paper, it barely talked about how do they get the matrix
# The size of the transition matrix should depend on the number of hidden states(N): H x H
# This part is in read.py
A = (27, 27)
###

### Next we need the start probability vector: pi
### A reasonable way is to assume all keys are equality likely 
### (Or we just pick letter t since word The has the largest frequency, also in read.py)

model.startprob_ = np.ones(27) / 27

### This is essentially all we need to train a HMM.

### One thing I am not sure about is the emission mean: read the plot_casino.py example
### That is basically saying, we need some prior knowledge about the observation and the hidden states
### Can we? For now lets assume the prior knowledge is that for each hidden state, the observation will be equally likely
### (which clearly not true) But I think it will be not mandatory 

# # gen_model.emissionprob_ = \
#     np.array([[1/K, 1/K, ... 1/K], (1)
#                ...                    
#               [1/K, 1/K, ... 1/K]) (27) 




### Following is how we actually gonna infer the hidden states

# split our data into training and validation sets (70/30 split)
split_index = int(data_matrix.shape[1] * 0.7)
X_train, X_validate= np.split(data_matrix, [split_index], axis=1)

# check optimal score
score = model.score(X_validate)

best_score = best_model = None
n_fits = 50
np.random.seed(13)
for idx in range(n_fits):
    model = hmm.CategoricalHMM(
        n_components=K, random_state=idx,
        init_params='se')  # don't init transition, set it below
    # we need to initialize with random transition matrix probabilities
    # because the default is an even likelihood transition
    # we know transitions are rare (otherwise the casino would get caught!)
    # so let's have an Dirichlet random prior with an alpha value of
    # (0.1, 0.9) to enforce our assumption transitions happen roughly 10%
    # of the time
    model.transmat_ = A
    model.fit(X_train)
    score = model.score(X_validate)
    print(f'Model #{idx}\tScore: {score}')
    if best_score is None or score > best_score:
        best_model = model
        best_score = score
        
# use the Viterbi algorithm to predict the most likely sequence of states
# given the model
states = best_model.predict(data_matrix)

# Finally, let's see if we can tell how the die is loaded.

print(f'Emission Matrix Generated:\n{model.emissionprob_.round(3)}\n\n'
      f'Emission Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n\n')

# https://hmmlearn.readthedocs.io/en/latest/tutorial.html#multiple-sequences