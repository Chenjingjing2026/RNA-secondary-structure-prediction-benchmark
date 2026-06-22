import numpy as np # Import the module
import DataReadClass as dr
import BasisFuncTionClass as bf
import os

def mainFunctionModel(PriTestSeqList, PriTestStructureList, PriTestLengthList, PriTestList, PriFileTestIndex):
    print("Start cal score")
    DataRead = dr.DataRead() # Definition object.
    # PriTestSeqList, PriTestStructureList, PriTestLengthList, PriTestList, PriFileTestIndex = DataRead.PriArchiveData() # Reading the real data.
    PreStructureTest = DataRead.PreStructureData() # Reading the prediction results.
    BasisFunction = bf.BasisFunction() # Definition object
    Index = 0
    Count = 0 # Record the number of RNA sequences
    AllPPV = 0 # Record the sum of PPV.
    AllSen = 0 # Record the sum of SEN.
    AllFscore = 0 # Record the sum of F-score.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    while(Index < len(np.array(PriTestList))):
        print('Cal Score loop index: ' + str(Index))
        PriStructure = [] # Restore the real structure.
        AllPreStructure = [] # Restore the prediction structure.
        AllPriList = [] # Restore the list of ct file.
        '''
            Remove the prediction results of padding bases and splicing the prediction 
            results of multiple sub-sequences. Restore the processed results in PriStructure、
            AllPreStructure、and AllPriList.
        '''
        if(Index != len(np.array(PriTestList))-1):
           if(PriFileTestIndex[Index] != PriFileTestIndex[Index + 1]):
               for i in range(PriTestLengthList[Index]):
                   PriStructure.append(PriTestStructureList[Index][i])
                   AllPreStructure.append(PreStructureTest[Index][i])
                   AllPriList.append(PriTestList[Index][i])
               Index += 1
           else:
               for i in range(200):
                   PriStructure.append(PriTestStructureList[Index][i])
                   AllPreStructure.append(PreStructureTest[Index][i])
                   AllPriList.append(PriTestList[Index][i])
               Index += 1
               while(Index < len(np.array(PriTestList)) and PriFileTestIndex[Index] == PriFileTestIndex[Index-1]):
                   if(Index != len(np.array(PriTestList))-1):
                       if(PriFileTestIndex[Index] == PriFileTestIndex[Index+1]):
                           for i in range(100,200):
                               PriStructure.append(PriTestStructureList[Index][i])
                               AllPreStructure.append(PreStructureTest[Index][i])
                               AllPriList.append(PriTestList[Index][i])
                       else:
                           for i in range(100, PriTestLengthList[Index]):
                               PriStructure.append(PriTestStructureList[Index][i])
                               AllPreStructure.append(PreStructureTest[Index][i])
                               AllPriList.append(PriTestList[Index][i])
                   else:
                        for i in range(100, PriTestLengthList[Index]):
                            PriStructure.append(PriTestStructureList[Index][i])
                            AllPreStructure.append(PreStructureTest[Index][i])
                            AllPriList.append(PriTestList[Index][i])
                   Index += 1
        else:
            for i in range(PriTestLengthList[Index]):
                PriStructure.append(PriTestStructureList[Index][i])
                AllPreStructure.append(PreStructureTest[Index][i])
                AllPriList.append(PriTestList[Index][i])
            Index += 1
        print("start conv")

        PreStructureList = BasisFunction.ConvergeToPri(np.array(AllPreStructure))  # Transform the prediction results into dot-bracket sequences.
        ResultStack,FirstLayerStack, SecondaryLayerStack, ThirdLarerStack= BasisFunction.CreateStructureStack(PreStructureList) # Get the PCRs for different sub-structure.
        Combination = BasisFunction.MainFuntion(FirstLayerStack, SecondaryLayerStack, ThirdLarerStack, PreStructureList, AllPriList) # Get the prediction results.
        Ppv = 0
        Sen = 0
        Fscore = 0
        MCC=0
        print('conv finish')
        for i in range(len(np.array(Combination))):
            PPV, Sensitive, Fs, mcc = BasisFunction.ValueOfPPVSen(AllPriList, Combination[i]) # Calculate the accuracy the the prediction results.
            if (Fs > Fscore):
                Sen = Sensitive
                Ppv = PPV
                Fscore = Fs
                MCC = mcc
                # 将结果写入文件（追加模式，自动刷新缓存）
                with open('/home/chenjingjing/Models/DMfold/Saver_Result/bprna_1m/results.txt', 'a') as file:
                    # file.write(f"{Count+1}, {Sen}, {Ppv}, {Fscore}\n")
                    file.write("{}, {}, {}, ,{}, {}\n".format(Count + 1, Sen, Ppv, Fscore, MCC))
    print('save socre Finish')
#
#     AllPPV += Ppv # Add the PPV
#     AllSen += Sen # Add the Sen
#     AllFscore += Fscore # Add the F-score
#     Count += 1
# print("$$$$$$$$$$$$$$$$$$$$$$$$$")
# print(AllPPV / Count) # Output the value of average PPV
# print(AllSen / Count) # Output the value of average SEN
# print(AllFscore/Count) # Output the value of average F-score.