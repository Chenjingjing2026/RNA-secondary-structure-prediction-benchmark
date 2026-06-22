import DataMerge as dm
import numpy as np
import shutil as st
'''
This code can divide all the cleaned raw data into Testset and Train_Validation set. Testset is used to test the performance
of the method. Train_Validation set  is used for the ten fold cross validation
'''
filePath = 'OriginalData/All_Clearn_Data'  # All_Clearn_Data stores all raw cleaning data.
def DataCreate():
    nameList = dm.EachFile(filePath) # Read all the files in filePath and return a list which stores all the name of raw data.
    nameSize = int(len(nameList) * 0.1) # Calculate the number of all original files and take 10% as the length of the test set.
    idx = np.random.choice(len(nameList), size=nameSize, replace=False) # Random acquisition of nameSize integers, which are between 0 and len(nameList).
    for i in range(len(nameList)):
        if(i in idx): # if i in the set of random number.
            ctPath = dm.FileNmae(filePath, nameList[i]) # Get the absolute path of the file.
            st.copy(ctPath, 'OriginalData/TestData') # Copy the file to TestData which stores all the test data.
        else:
            ctPath = dm.FileNmae(filePath, nameList[i]) # if i not in the random number.
            st.copy(ctPath, 'OriginalData/Train_Validation_Data') # Copy the file to Train_Validation_Data which stores all the train and validation data.
DataCreate()