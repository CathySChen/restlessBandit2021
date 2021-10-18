'''This code fit an HMM model to choice sequences (2-armed restless bandit) for a single session. The HMM model is 1. parameter-tied 2. reseeded (random initial T matrix) 3. fixed emission matrix, 5. fixed initial distribution. 6. optimization based by observed log likelihood and side criteria is Tmatrix[0,0] and [1,1]> 0.5. This code is created by Cathy Chen. Date 05/2021.'''
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
        #print (alpha)
        beta = backward(V, a, b)
        
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
        ### xi is 3x3x299
        
        ## gamma is a 3x300 matrix
        gamma = np.sum(xi, axis=1)
        
        ## tmp is transition matrix (a) in log likelihood form
        
        #### parameter tying
        tmp = np.sum(xi,2)
        
        # Add additional T'th element in gamma
        #gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        #K = b.shape[1]
        #denominator = np.sum(gamma, axis=1)
        #for l in range(K):
         #   b[:, l] = np.sum(gamma[:, V == l], axis=1)
 
        #b = np.divide(b, denominator.reshape((-1, 1)))
        
        ### fix b, b stays the same
        #b = np.array([[1/2,1/2],[1,0],[0,1]])
    return tmp 
        
def parameterTying (tmp):
    tmp[0,1] = (tmp[0,1] + tmp[0,2])/2
    tmp[0,2] = tmp[0,1]

    tmp[1,1] = (tmp[1,1]+tmp[2,2])/2
    tmp[2,2] = tmp[1,1]

    tmp[1,0] = (tmp[1,0]+tmp[2,0])/2
    tmp[2,0] = tmp[1,0]
    
    a = tmp/np.sum(tmp,1).reshape((-1, 1))
    
    return a
                

def optimizeLog (V, a, b, initial_distribution):
    T = V.shape[0] ## 300 - time.trial
    M = a.shape[0] ## number of states - 3
    
    ## p (state)
    omega = np.zeros((T, M))
    omega[0, :] = (initial_distribution) #inital state distribution
   
    for i in range (1,T):      
        for j in range (0,M):
            omega[i,j] = np.dot((a[:,j]),(omega[(i-1)]))
    #print (omega)
    ## p (choice) =sum( p(choice|state) *p(state))
    probList = []
    for k in range (0,T):
        pchoice = np.dot(b[:,V[k]],omega[k])
        probList.append(pchoice)
        
    logLike = (np.sum(np.log(probList)))*(-1)
    
    ### this is the observed log likelihood
    
    return logLike   
    


def viterbi(V, a, b, initial_distribution):
    T = V.shape[0] ## 300 - time.trial
    M = a.shape[0] ## number of states - 3

    omega = np.zeros((T, M))
    #omega[0, :] = np.log(initial_distribution * b[:, V[0]]) # prior
    omega[0, :] = (initial_distribution * b[:, V[0]]) # prior

    prev = np.zeros((T - 1, M)) 
 
    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
            
    
 
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)
 
            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
    

    ## shape of omega is 300x3
 
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
    
    exploreList = []
    reseed = 10
    
    folder = input ('Please input data directory: ')
    
    optimAList = [[0]]
    
    for j in range (1,2):###animal ID
        for s in range (1,2):  ### session
            df = pd.read_csv(folder + '//session' + str(s) + '//'+str(j) + '.csv', skiprows = 0)

            choice = df['choice chosen'].values - 1
            #newChoice = []
            #for i in range (len(choice)):
             #   newChoice.append(int(choice[i])-1)

            #newChoice = np.asarray(newChoice)

            ## emission probs (p(choice|state))
            b = np.array([[1/2,1/2],[1,0],[0,1]])

            # initial prob distribution
            initial_distribution = np.array((1,0,0))


            '''HMM training using Baum-Welch algorithm. Return transition probabilities A and emission probabilities B'''

            count = 0
            while True:
                count += 1
                optimAlist = []
                obsLogList = []

                for h in range (reseed):  
                    #### randomly reseed transition matrix A 
                    rand1 = np.random.random()
                    rand2 = np.random.random()
                    a = np.array([[(1-rand1),rand1/2,rand1/2],[1-rand2,rand2,0,],[1-rand2,0,rand2]])

                    hmm=baum_welch(choice,a,b,initial_distribution, n_iter =20) #matrix A and B
                
                    matrixA = parameterTying (hmm)
                    optimAlist.append(matrixA)

                    obsLog = optimizeLog (choice, matrixA, b, initial_distribution)
                    obsLogList.append(obsLog)

                optimA = optimAlist[obsLogList.index(min(obsLogList))]

                if optimA[1,1] > 0.5 and optimA[0,0]>0.5 or count == 10:
                    break
            print (optimA)
                
            stateLabel = viterbi(choice,optimA,b,initial_distribution)

            labelList = stateLabel.tolist()

            pExplore = labelList.count(0)/300
            exploreList.append(pExplore)
            print ("pexplore for " + str(j) + ' is: ' + str(pExplore))
            
            Dict = { 'state': labelList}
            df = pd.DataFrame(Dict)
            df.to_csv('HMM_session'+ str(s)+ '_animal'+str(j) +'genie.csv')
            
    print ('All done!')

    
if __name__ == "__main__":
    main()
             