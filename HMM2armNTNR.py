'''This code fit an HMM model to a single-session of choice sequence (2-armed restless bandit). The HMM model is 1. not parameter-tied 2. no reseeding 3. randomized initiation T matrix 4. emission matrix is optimized (not fixed), 5. fixed initial distribution. This code is created by Cathy Chen. Date 01/2021.'''

import numpy as np
import pandas as pd


def forward (V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0],a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
    return alpha

def backward (V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
 
    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
 
    return beta

def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)
 
    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)
 
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
 
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)
 
        b = np.divide(b, denominator.reshape((-1, 1)))
 
    return {"a":a, "b":b}


def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]
 
    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])
 
    prev = np.zeros((T - 1, M))
 
    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
 
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)
 
            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
 
    # Path Array
    S = np.zeros(T)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # Flip the path array since we were backtracking
    result = np.flip(S, axis=0)
    
 
    return result

def main ():

    folder = input ('Please input data directory: ')
    
    exploreList = []
    
    for j in range (1,2):
        df = pd.read_csv(foler + '//' + str(j) + '.csv', skiprows = 1)
        choice = df['choice'].tolist()
        newChoice = []
        for i in range len(choice):
            newChoice.append(int(choice[i])-1)
        
        print (newChoice)
        
        '''HMM training using Baum-Welch algorithm. Return transition probabilities A and emission probabilities B'''

        ## transition probs
        rand1 = np.random.random()
        rand2 = np.random.random()
        a = np.array([[(1-rand1),rand1/2,rand1/2],[1-rand2,rand2,0],[1-rand2,0,rand2]])


        ## emission probs (p(choice|state))
        b = np.array([[1/2,1/2],[1/2,0],[0,1/2]])

        # initial prob distribution
        initial_distribution = np.array((1/3,1/3,1/3))

        hmm = baum_welch(np.asarray(newChoice),a,b,initial_distribution, n_iter =100)

        print (hmm['a'])

        stateLabel = viterbi(np.asarray(newChoice),hmm['a'],hmm['b'],initial_distribution)

        #print (stateLabel)
        labelList = stateLabel.tolist()
        
        pExp = labelList.count (0)/len(labelList)
        exploreList.append(pExp)
        print (pExp)
    Dict = { 'explore': exploreList}
    df = pd.DataFrame(Dict)
    df.to_csv('HMM.csv')

    
if __name__ == "__main__":
    main()
             