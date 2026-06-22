import DataMerge as dm
import numpy as np
import shutil as st
import os
'''
Obtain trainset and validationset based on ten-fold validation data. 
Random select  nine of validation data as trainset and the remaining one as validationset.
'''
prePath = 'OriginalData/Ten_Fold_Validation' # prepath is the path of  all the fold of data.
def DataCreate():
    for i in range(10):
        name = "TrainSet_" + str(i + 1)
        address = 'OriginalData/TrainData/' + name
        os.makedirs(address) # Create the file name for each trainset.
        for j in range(10):
            if(j != i):
                filePath = prePath+ "/" + "Validation" + str(j + 1)
                nameList = dm.EachFile(filePath)
                nameSize = len(nameList)
                idx = np.random.choice(len(nameList), size=nameSize, replace=False)
                for m in range(len(nameList)):
                    if (m in idx):
                        ctPath = dm.FileNmae(filePath, nameList[m]) # Copy all files in filePath to address.
                        st.copy(ctPath, address)
DataCreate()