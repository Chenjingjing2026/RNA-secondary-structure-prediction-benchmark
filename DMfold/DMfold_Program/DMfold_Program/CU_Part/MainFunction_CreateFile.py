import numpy as np  # Import the module
import DataReadClass as dr
import BasisFuncTionClass as bf
DataRead = dr.DataRead()# Definition object.
PriTestSeqList, PriTestStructureList, PriTestLengthList, PriTestList, PriFileTestIndex = DataRead.PriArchiveData() # Reading the real data.
PreStructureTest = DataRead.PreStructureData()# Reading the prediction results.
BasisFunction = bf.BasisFunction()# Definition object
Index = 0
Count = 0 # Record the number of RNA sequences
AllPPV = 0 # Record the sum of PPV.
AllSen = 0 # Record the sum of SEN.
AllFscore = 0 # Record the sum of F-score.
while(Index < len(np.array(PriTestList))):
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
    Pri = np.argmax(PriStructure, 1)
    Pre = np.argmax(AllPreStructure, 1)
    PreStructureList = BasisFunction.ConvergeToPri(np.array(AllPreStructure))  # Transform the prediction results into dot-bracket sequences.
    ResultStack, FirstLayerStack, SecondaryLayerStack, ThirdLarerStack = BasisFunction.CreateStructureStack(PreStructureList)  # Get the PCRs for different sub-structure.
    Combination = BasisFunction.MainFuntion(FirstLayerStack, SecondaryLayerStack, ThirdLarerStack, PreStructureList,AllPriList)  # Get the prediction results.
    Ppv = 0
    Sen = 0
    Fscore = 0
    ctName = str(Count) + 'th.ct'
    f_ct = open('Result/CT_Result/' + ctName, 'w+') # Open the ct file
    fastaName = str(Count) + 'th.fasta'
    f_fasta = open('Result/Fasta_Result/' + fastaName, 'w+') # Open the fasta file.
    for i in range(len(np.array(Combination))):
        f_ct.write("The " + str(i+1) + 'th structure.') # Write the name of prediction structure.
        f_ct.write('\n')
        f_fasta.write("The " + str(i+1) + 'th structure.')# Write the name of prediction structure.
        f_fasta.write('\n')
        List = [] # Restore the prediction results.
        for jj in range(len(np.array(AllPriList))):
            Tem = []
            Tem.append(str(AllPriList[jj][0]))
            if(AllPriList[jj][1] == 1):
                Tem.append('A')
            elif(AllPriList[jj][1] == 2):
                Tem.append('U')
            elif (AllPriList[jj][1] == 3):
                Tem.append('G')
            else:
                Tem.append('C')
            Tem.append(str(0))
            List.append(Tem)
        List = np.array(List) # Transform the format of List into np.array.
        for stem in range(len(np.array(Combination[i]))):
            PreFirst = Combination[i][stem][0]
            PreSecondary = Combination[i][stem][1]
            EarFirst = Combination[i][stem][2]
            EarSecondary = Combination[i][stem][3]
            while(PreFirst <= PreSecondary): # Create the ct structure file.
                List[PreFirst][2] = str(EarSecondary + 1)
                List[EarSecondary][2] = str(PreFirst + 1)
                PreFirst = PreFirst + 1
                EarSecondary = EarSecondary - 1
        for j in range(len(List)): # Write the ct file.
            jointsFrame = List[j]
            for m in range(len(List[j])):
                strNum = str(List[j][m])
                f_ct.write(strNum)
                f_ct.write("   ")
            f_ct.write('\n')
        FastaList = []
        for iii in range(len(List)):
            Tem = []
            for jjj in range(len(List[iii])):
                Tem.append(List[iii][jjj])
            Tem.append('0')
            FastaList.append(Tem)
        FastaList = BasisFunction.Dot_Bracket(FastaList) # Transform the format into dot_bracket.
        RNASequence = ''
        DotSequence = ''
        for iiii in range(len(np.array(FastaList))):
            RNASequence = RNASequence + FastaList[iiii][1]
            if(FastaList[iiii][3] != '0'):
                DotSequence = DotSequence + FastaList[iiii][3]
            else:
                DotSequence = DotSequence + '.'
        f_fasta.write(RNASequence) # Write the fasta structure.
        f_fasta.write('\n')
        f_fasta.write(DotSequence)
        f_fasta.write('\n')
