import DataMerge as dm
import numpy as np
import shutil as st
import os
'''
Train_Validation_Data can be randomly divided into 10 copies. Each of copy represents a fold of data.
'''
filePath = 'OriginalData/Train_Validation_Data' # The path of all Train_Validation_Data.
def DataCreate():
    nameList = dm.EachFile(filePath) # Obtain all the name of train and validation data.
    nameSize = int(len(nameList) * 0.1) # Obtain the length of a fold validation data.
    for j in range(10): # The length of top nine folds validation data is nameSize and the length of tenth fold validation data is all the surplus data of Train_Validation_Data.
        if(j!= 9):
            List = dm.EachFile(filePath) # Calculate the current data length in filePath.
            idx = np.random.choice(len(List), size=nameSize, replace=False) # Random acquisition of nameSize random numbers, which are between 0 and len(List).
            name = "Validation" + str(j + 1)
            address = 'OriginalData/Ten_Fold_Validation/' + name
            os.makedirs(address) # Create folder to store the validation data.
            for i in range(len(List)):
                if(i in idx):
                    ctPath = dm.FileNmae(filePath, List[i]) # Get the absolute of the file which meeting conditions.
                    st.move(ctPath, address) # Move the file to address.
        else: # When i = 9 means that the fold is the tenth fold validation.
            List = dm.EachFile(filePath)
            Size = len(List) # Calculate the current data length in filePath.
            name = "Validation" + str(j + 1)
            address = 'OriginalData/Ten_Fold_Validation/' + name
            os.makedirs(address)
            idx = np.random.choice(len(List), size=Size, replace=False)
            for i in range(len(List)):
                if(i in idx):
                    ctPath = dm.FileNmae(filePath, List[i]) # Move all remaining data from Train_Validation_Data to address.
                    st.move(ctPath, address)
DataCreate()