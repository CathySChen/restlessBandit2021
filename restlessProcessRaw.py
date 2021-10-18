import os
import os.path
import csv

    
def main():
    """Reads a file of raw data and analyze side bias"""
    print ("This program analyzes win shift data for Bandit test")
    #folder=r"C:\Users\grissomlab\Desktop\1003"
    folder=input("please enter the directory name:")
    destinationFolder=input("Please enter the destination folder name: ")
    
    ## Correct -> high prob payoff  -> image 1 + win
    ## Incorrect bad feedback  -> low prob payoff -> image 2 + win
    ## Incorrect -> low prob no payoff  -> image 2+ lose
    ## Correct_ bad feedback  -> high prob no payoff -> image 1 +lose
    
    filenameList=[]
    #nameList=['eggplant1','eggplant2','eggplant3','eggplant4','fig1','fig2','fig3','fig4','ginger1','ginger2','ginger3','ginger4','habanero1','habanero2','habanero3','habanero4','ugni1','ugni2','ugni3','ugni4','wasabi1','wasabi2','wasabi3','wasabi4','yam1','yam2','yam3','yam4','zucchini1','zucchini2','zucchini3','zucchini4']
    nameList = ['genie1']
    
   
    
    for file in os.listdir(folder):
        filename = os.path.join(folder, file)
  
        with open(filename) as csvfile:
            csvReader = csv.reader(csvfile, delimiter=',')
            
            
            lineCount=0
            leftValueList = []
            rightValueList = []
            choiceList = []
            rewardList = []
            for line in csvReader:
                lineCount+=1
                
                if lineCount <= 10 or 12<=lineCount<=20:
                    continue
                elif lineCount == 11:
                    name=line[1]
                    ID = nameList.index(name) + 1

                else:
                    item=line[3]
                    if item == "Reward_value_R":
                        rightValueList.append(int(line[8]))
                        
                    if item == "Reward_value_L":
                        leftValueList.append(int(line[8]))
                        
                    if item == "L side touched_ reward":
                        choiceList.append(1)
                        rewardList.append(1)
                        
                    if item == "R side touched_ reward":
                        choiceList.append(2)
                        rewardList.append(1)
                        
                    if item == "L side touched_ no reward":
                        choiceList.append(1)
                        rewardList.append(0)
                        
                    if item == "R side touched_ no reward":
                        choiceList.append(2)
                        rewardList.append(0)
                        
            leftValueList = leftValueList[1:]
            rightValueList = rightValueList[1:]

            destination_file = os.path.join(destinationFolder, str(ID)+".csv")
            with open(destination_file, 'w',newline='') as outputcsvfile:
                csvWriter=csv.writer(outputcsvfile)
                csvWriter.writerow([name])
                csvWriter.writerow(['left probability','right probability','choice chosen','reward outcome'])          
                for i in range (len(choiceList)):
                    csvWriter.writerow([leftValueList[i],rightValueList[i],choiceList[i],rewardList[i]])
    print ("Done! The output files are in the designated folder.")

                
if __name__ == "__main__":
    main()
