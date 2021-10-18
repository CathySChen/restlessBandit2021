import numpy as np
import csv
import os
import os.path
import pandas as pd

def main():
    
    IDList = []
    MIlist = []
    
    for number in range (1,33):
        with open(str(number)+'.csv', newline='') as csvfile:
            csvReader = csv.reader(csvfile, delimiter=',')
            
            lineCount = 0 
            sideList = []
            rewardList = []
            IDList.append(number)
            
            for line in csvReader:
                lineCount+=1
                
                if lineCount == 1:
                    continue
                    
                else:
                    side = int(line[4])
                    ## may need to change this
                    sideList.append(side)
                    ## and may need to change this
                    reward = int (line [5])
                    rewardList.append (reward)
                
            currentList = sideList [1:] #current
            lastList = sideList [:-1] # previous
            rewardList = rewardList [:-1]
            
            total = len(currentList)
            reward = rewardList.count (1)
            NR = rewardList.count (0)
            #print (NR)

   
            PX = np.array([1,2]) #previous
            PY = np.array ([1,2]) #current
            PZ = np.array ([0,1]) #reward

            accum = np.zeros((2, 2, 2)) #height, row, col

            for n in range (len(lastList)):

                indexX = list(PX).index (lastList[n])
                indexY = list(PY).index (currentList[n])
                indexZ = list(PZ).index (rewardList[n])

                accum[indexZ,indexY,indexX] += 1

            Pheight = [np.sum(accum[0]), np.sum(accum[1])] # Z reward
            Pcol = np.asarray(accum).sum(axis=1) # XZ previous +reward
            Prow = np.asarray(accum).sum(axis=2) # YZ current + reward
            
            #print (total)


            MI = 0
            for height in range (0,2): #reward
                for row in range (0,2): #current
                    for col in range (0,2): #previous
                        if accum[height][row][col] == 0:
                            MI += 0 
                        else:
                            MI += (accum[height][row][col]/total)*(np.log2(((accum[height][row][col]/total)*(Pheight[height]/total)/((Pcol[height][col]/total)*(Prow[height][row]/total)))))

            MIlist.append(MI)
            #print (MIlist)
            
    df = pd.DataFrame({'MI': MIlist})
    df.to_csv('mutual information.csv', index=False)

                                                
if __name__ == "__main__":
    main()
