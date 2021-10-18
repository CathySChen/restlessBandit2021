'''This code takes in a csv with two columns: HMM optimized transition matrix [0,0] and [1,1], and calculates the energy barrier, depth difference between two states, and a scaling factor. This code is created by Cathy Chen, Date: 05/2021'''

import pandas as pd
import numpy as np
import scipy.linalg
import os

def main():

    folder = input ('Please input data directory: ')

    df = pd.read_csv(folder + '\\transition matrix1.csv')

    tore = df['tore'].values
    toit = df['toit'].values

    depthList = []
    barrierList = []
    readiness = []

    for i in range (0,210):

        #tmatrix = np.transpose(np.array([[float(tore[i]),1-float(tore[i])],
         #   [1-float(toit[0]),float(toit[0])]]))
        if tore[i] != 'na':

            tmatrix = np.array([[float(tore[i]),1-float(tore[i])],
                [1-float(toit[0]),float(toit[0])]])

            values, left= scipy.linalg.eig(tmatrix, right = False, left = True)


            index = list(np.round (values.real.tolist(),1)).index(1)
            #index  = 1

            leftVec = left[:,index]

            statProb = leftVec/np.sum(leftVec)


            ### E_oit - E_ore, should be a positive number
            depthDiff = np.log (statProb[0]/statProb[1])


            ### energy barrier for explore
            barrier = -1 * np.log (1-float(tore[i]))

            A = (statProb[0]/statProb[1])/((1-float(tore[i]))/(1-float(toit[i])))

            depthList.append(depthDiff)
            barrierList.append(barrier)
            readiness.append(A)

        else:
            depthList.append('na')
            barrierList.append('na')
            readiness.append('na')


        dataDict = { 'depth difference':depthList,'explore activation barrier':barrierList, 'transformation readiness parameter': readiness}
        df2 = pd.DataFrame(dataDict)
        df2.to_csv(os.path.join('energy landscape.csv'))






     
if __name__ == "__main__":
    main() 