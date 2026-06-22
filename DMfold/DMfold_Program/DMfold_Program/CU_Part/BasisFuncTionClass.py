import numpy as np
import tensorflow as tf
import itertools as it
import copy as cy # Import the module.
class BasisFunction:
    def ConvergeToPri(self,  PreStructure):  # Transform the prediction results into dot-bracket sequences.
        PreStructure = tf.argmax(PreStructure, 1) # Get the maximum position of each line
        sess = tf.Session()
        PreStructure = sess.run(PreStructure)
        PreStructureList = []
        for i in range(len(PreStructure)):
            if (PreStructure[i] == 0):
                PreStructureList.append(1)
            elif (PreStructure[i] == 6):
                PreStructureList.append(2)
            elif (PreStructure[i] == 1):
                PreStructureList.append(3)
            elif (PreStructure[i] == 5):
                PreStructureList.append(4)
            elif (PreStructure[i] == 2):
                PreStructureList.append(5)
            elif (PreStructure[i] == 4):
                PreStructureList.append(6)
            else:
                PreStructureList.append(0)
        return PreStructureList
    def CreateStructureStack(self, PreStructure):  # According to the pseudoknot-free sub-structure,convert prediction results to PCRs.
        flag = False
        ResultStack = []  # Restore the PCRs
        FirstLayerStack = []
        SecondaryLayerStack = []
        ThirdLarerStack = []
        MedStack = []
        count = -1
        for i in range(len(PreStructure)):
            if (PreStructure[i] != 0):
                if (flag == False):
                    MedStack.append(i)
                    flag = True
                if (i == len(PreStructure) - 1):
                    count += 1
                    MedStack.append(i)
                    MedStack.append(MedStack[1] - MedStack[0] + 1)
                    MedStack.append(PreStructure[i])
                    if(len(np.array(ResultStack)) != 0):
                        if(PreStructure[i] == ResultStack[-1][3]):
                            MedStack.append(MedStack[0] - ResultStack[-1][1]-1)
                        else:
                            MedStack.append(-1)
                    else:
                        MedStack.append(-1)
                    MedStack.append(0)
                    MedStack.append(count)
                    MedStack.append(0)
                    ResultStack.append(MedStack)
                    MedStack = []
                    flag = False
                if(PreStructure[i] != PreStructure[i-1] and i>=1 and PreStructure[i-1] != 0):
                    count += 1
                    MedStack.append(i - 1)
                    MedStack.append(MedStack[1] - MedStack[0] + 1)
                    MedStack.append(PreStructure[i - 1])
                    if (len(np.array(ResultStack)) != 0):
                        if (PreStructure[i - 1] == ResultStack[-1][3]):
                            MedStack.append(MedStack[0] - ResultStack[-1][1] - 1)
                        else:
                            MedStack.append(-1)
                    else:
                        MedStack.append(-1)
                    MedStack.append(0)
                    MedStack.append(count)
                    MedStack.append(0)
                    ResultStack.append(MedStack)
                    MedStack = []
                    MedStack.append(i)
                    flag = True
            else:
                if (flag):
                    count += 1
                    MedStack.append(i - 1)
                    MedStack.append(MedStack[1] - MedStack[0] + 1)
                    MedStack.append(PreStructure[i-1])
                    if (len(np.array(ResultStack)) != 0):
                        if (PreStructure[i-1] == ResultStack[-1][3]):
                            MedStack.append(MedStack[0] - ResultStack[-1][1] - 1)
                        else:
                            MedStack.append(-1)
                    else:
                        MedStack.append(-1)
                    MedStack.append(0)
                    MedStack.append(count)
                    MedStack.append(0)
                    ResultStack.append(MedStack)
                    MedStack = []
                    flag = False
        for i in range(2):
            indexCount = 0
            for j in range(len(np.array(ResultStack))):
                if(i == 0):
                    if(ResultStack[j][3] == 1 or ResultStack[j][3] == 2):
                        FirstLayerStack.append(cy.deepcopy(ResultStack[j]))
                        FirstLayerStack[-1][7] = indexCount
                        indexCount += 1
                elif(i == 1):
                    if (ResultStack[j][3] == 3 or ResultStack[j][3] == 4):
                        SecondaryLayerStack.append(cy.deepcopy(ResultStack[j]))
                        SecondaryLayerStack[-1][7] = indexCount
                        indexCount += 1
                else:
                    if (ResultStack[j][3] == 5 or ResultStack[j][3] == 6):
                        ThirdLarerStack.append(cy.deepcopy(ResultStack[j]))
                        ThirdLarerStack[-1][7] = indexCount
                        indexCount += 1
        '''
        All the PCRs are in ResultStack. The first class of PCRs in the FirstLayerStack. The secondary class of PCRs in the  SecondaryLayerStack.
        The third class of PCRs in the ThirdLarerStack.
        '''
        return ResultStack, FirstLayerStack, SecondaryLayerStack, ThirdLarerStack

    def MatrixCreate(self, PriList, PreStructure):  # Create the matrix to restore all the longest stem area.
        Matrix = np.zeros(shape=[len(PriList), len(PriList)])  # Create a two-dimensional matrix, the length and width of the matrix are the same as the length of the PriList, and initialize all values of the matrix to 0, 0 means unpaired.
        for i in range(len(PriList)):
            for j in range(i, len(PriList)):
                # Since the four bases A, U, G, and C of RNA are made to 1, 2, 3, and 4 in PriList, and A-U, G-C, and G-U can be paired, the absolute value difference between the triangles in the matrix is firstly listed. Marked as 1 for the base pair.
                if (abs(PriList[i][1] - PriList[j][1]) == 1 and abs(i - j) - 1 >= 2):
                    Matrix[i][j] = 1
        # Because U-G pairing is unconventional pairing, U-G pairing should be breaken at the boundary of stem areas.
        for i in range(len(PriList)):
            for j in range(i, len(PriList)):
                if (Matrix[i][j] == 1 and ((PriList[i][1] == 2 and PriList[j][1] == 3) or (PriList[i][1] == 3 and PriList[j][1] == 2))): # 当前值为1，且行列值为2和3时候进入处理。
                    if (i == 0 or j == len(PriList) - 1 or (j - 1) <= i + 1):
                        Matrix[i][j] = 0
                    else:
                        if (Matrix[i - 1][j + 1] == 0 or Matrix[i + 1][j - 1] == 0):
                            Matrix[i][j] = 0
                    if ((PreStructure[i] == 1 and PreStructure[j] == 2) or ((PreStructure[i] == 2 and PreStructure[j] == 1))):
                        Matrix[i][j] = 1
        return Matrix
    def MaxStem(self, i, j, p, q, Matrix):  # This method is used to get all stems in a certain area in matrix.
        AllSterm = []  # Used to restore the stems: [i, j, p, q]
        IndexQ = q
        while (IndexQ >= p):
            if (IndexQ - p >= j - i):
                IndexI = i
                IndexJ = j
                IndexP = IndexQ - (IndexJ - IndexI)
            else:
                IndexI = i
                IndexP = p
                IndexJ = IndexI + IndexQ - IndexP
            count = 0
            MedQ = IndexQ
            MedI = IndexI
            Indexj = IndexJ
            flag = False
            StemList = []
            while (MedQ >= IndexP and MedI <= Indexj):
                if (Matrix[MedI][MedQ] == 1):
                    count += 1
                    if (MedQ == IndexP and MedI == Indexj):
                        if (flag == False):
                            StemList.append(MedI)
                            StemList.append(MedI)
                            StemList.append(MedQ)
                            StemList.append(MedQ)
                        else:
                            StemList.insert(1, MedI)
                            StemList.insert(2, MedQ)
                        StemList.insert(4, count)
                        AllSterm.append(StemList)
                        StemList = []
                        count = 0

                    else:
                        if (flag == False):
                            StemList.append(MedI)
                            StemList.append(MedQ)
                            flag = True
                else:
                    if (flag == True):
                        StemList.insert(1, MedI - 1)
                        StemList.insert(2, MedQ + 1)
                        flag = False
                        StemList.insert(4, count)
                        AllSterm.append(StemList)
                        StemList = []
                        count = 0
                MedQ -= 1
                MedI += 1
            IndexQ -= 1
        IndexI = i + 1
        while (IndexI <= j):
            if (q - p >= j - IndexI):
                IndexJ = j
                IndexQ = q
                IndexP = IndexQ - (IndexJ - IndexI)
            else:
                IndexQ = q
                IndexP = p
                IndexJ = IndexI + IndexQ - IndexP
            count = 0
            MedQ = IndexQ
            MedI = IndexI
            Indexj = IndexJ
            flag = False
            StemList = []
            while (MedQ >= IndexP and MedI <= Indexj):
                if (Matrix[MedI][MedQ] == 1):
                    count += 1
                    if (MedQ == IndexP and MedI == Indexj):
                        if (flag == False):
                            StemList.append(MedI)
                            StemList.append(MedI)
                            StemList.append(MedQ)
                            StemList.append(MedQ)
                        else:
                            StemList.insert(1, MedI)
                            StemList.insert(2, MedQ)
                        StemList.insert(4, count)
                        AllSterm.append(StemList)
                        StemList = []
                        count = 0
                    else:
                        if (flag == False):
                            StemList.append(MedI)
                            StemList.append(MedQ)
                            flag = True

                else:
                    if (flag == True):
                        StemList.insert(1, MedI - 1)
                        StemList.insert(2, MedQ + 1)
                        flag = False
                        StemList.insert(4, count)
                        AllSterm.append(StemList)
                        StemList = []
                        count = 0
                MedQ -= 1
                MedI += 1
            IndexI += 1
        if(i == j and p == q):
            flag = True
        else:
            flag = False
        return AllSterm,  flag
    def SelectStem(self, AllStem, flag):  # Select the stems that meet the criteria and sort by the length of the stem.
        SelectStemList = []
        for i in range(len(np.array(AllStem))):
            if (flag):
                count = 1
            else:
                count = 2
            if (AllStem[i][4] >= count and AllStem[i][2] - AllStem[i][1] > 3):  # Select stem regions longer than the Count interval by more than 3 bases.
                SelectStemList.append(AllStem[i])
        SelectStemList = np.array(SelectStemList)
        if (len(np.array(SelectStemList)) != 0): # Sort by stem length
            SelectStemList = SelectStemList[np.lexsort(-SelectStemList.T)]
        return SelectStemList
    def CombinationSterm(self, layerSterm):
        Combination = []
        Med = []
        for i in range(len(np.array(layerSterm))):
            listnum = list(it.combinations(layerSterm, i+1))
            List = []
            for j in range(len(np.array(listnum))):
                List.append(list(listnum[j]))
            if((i+1) > 1):
                for p in range(len(np.array(List))):
                    flag = False
                    for q in range(i+1):
                        for r in range(q+1, i+1):
                            if(List[p][q][3] < List[p][r][0] or List[p][r][3] < List[p][q][0] or (List[p][q][1]<List[p][r][0] and List[p][r][3]<List[p][q][2]) or (List[p][r][1]<List[p][q][0] and List[p][q][3]<List[p][r][2])):
                                continue
                            else:
                                flag = True
                                break
                        if(flag):
                            break
                    if(flag):
                        continue
                    else:
                        Med.append(List[p])
            else:
                for p in range(len(np.array(List))):
                    Med.append(List[p])
        numList = []
        for i in range(len(np.array(Med))):
            num = 0
            for j in range(len(np.array(Med[i]))):
                num += Med[i][j][4]
            numList.append(num)
        largeNum = numList[0]
        for i in range(len(np.array(numList))):
            if(numList[i] > largeNum):
                largeNum = numList[i]
        for i in range(len(np.array(numList))):
            if(numList[i] == largeNum):
                Combination.append(Med[i])
        return Combination
    '''
    The method of LayerStermCom is the principle of CSCP. Used to obtain the optimal stem combinarion.
    '''
    def LayerStermCom(self, SelectStemListSort, Combination):   #  Combination may be NULL.
        Position = 0
        LayerSterm = []
        while(Position<len(SelectStemListSort)):
            num = SelectStemListSort[Position][4]
            stem = []
            while(Position<len(SelectStemListSort)):
                if(SelectStemListSort[Position][4] == num):
                    stem.append(SelectStemListSort[Position])
                else:
                    LayerSterm.append(stem)
                    break
                Position += 1
                if(Position == len(SelectStemListSort)):
                    LayerSterm.append(stem)
        LayerSterm = np.array(LayerSterm)
        for layer in range(len(LayerSterm)):
            NewComSterm = []
            layerSterm = LayerSterm[layer]
            if(len(np.array(Combination)) == 0):
                Combination = self.CombinationSterm(layerSterm)
            else:
                Combination = np.array(Combination)
                Combination = Combination.tolist()
                for i in range(len(np.array(Combination))):
                    Temlist = []
                    for j in range(len(np.array(layerSterm))):
                        flag = False
                        for p in range(len(np.array(Combination[i]))):
                            if(Combination[i][p][3] < layerSterm[j][0] or layerSterm[j][3] < Combination[i][p][0] or (Combination[i][p][1]<layerSterm[j][0] and layerSterm[j][3]<Combination[i][p][2]) or (layerSterm[j][1]<Combination[i][p][0] and Combination[i][p][3]<layerSterm[j][2])):
                                continue
                            else:
                                flag = True
                                break
                        if(flag == False):
                            Temlist.append(layerSterm[j])
                    if(len(np.array(Temlist)) == 0):
                        NewComSterm.append(Combination[i])
                    else:
                        Tem = self.CombinationSterm(Temlist)
                        for num in range(len(np.array(Tem))):
                            TemCom = cy.deepcopy(Combination[i])
                            for nn in range(len(np.array(Tem[num]))):
                                TemCom.append(Tem[num][nn])
                            NewComSterm.append(cy.deepcopy(TemCom))
                numList = []
                for i in range(len(np.array(NewComSterm))):
                    num = 0
                    for j in range(len(np.array(NewComSterm[i]))):
                        num += NewComSterm[i][j][4]
                    numList.append(num)
                largeNum = numList[0]
                for i in range(len(np.array(numList))):
                    if (numList[i] > largeNum):
                        largeNum = numList[i]
                Combination = []
                for i in range(len(np.array(numList))):
                    if (numList[i] == largeNum):
                        Combination.append(NewComSterm[i])
        return Combination
    def RateOfSterm(self, LocalOne, LocalTwo, OptimalStemList, PreStructure):  # Calculate the rate of PCRs
        RateOne = 0
        RateTwo = 0
        for i in range(len(np.array(OptimalStemList))):
            for j in range(int(OptimalStemList[i][0]), int(OptimalStemList[i][1]) + 1):
                if (j >= LocalOne[0] and j <= LocalOne[1] and (PreStructure[j] == 1 or PreStructure[j] == 3 or PreStructure[j] == 5)):
                    RateOne += 1
        for i in range(len(np.array(OptimalStemList))):
            for j in range(int(OptimalStemList[i][2]), int(OptimalStemList[i][3]) + 1):
                if (j >= LocalTwo[0] and j <= LocalTwo[1] and (PreStructure[j] == 2 or PreStructure[j] == 4 or PreStructure[j] == 6)):
                    RateTwo += 1
        return (RateTwo / LocalTwo[2] + RateOne / LocalOne[2]) / 2
    def MatrixStermCreate(self, ResultStack, Matrix, PreStructureList): # This method is used to restore the PCRs, which are a single rate is 1.
        TwoSuccess = []  #
        for i in range(len(np.array(ResultStack))):
            if(ResultStack[i][3] == 1):
                continue
            else:
                IndexQ = ResultStack[i][1]
                IndexP = ResultStack[i][0]
                for j in range(i):
                    if (ResultStack[j][3] == 1):
                        IndexI = ResultStack[j][0]
                        IndexJ = ResultStack[j][1]
                        AllSterm, flag = self.MaxStem(IndexI, IndexJ, IndexP, IndexQ, Matrix) # Obtain all stem areas in a certain area。
                        SelectStemList = self.SelectStem(AllSterm,  flag) # Selecting stem regions that meet the conditions
                        if (len(SelectStemList) != 0):
                            OptimalStemList = self.LayerStermCom(SelectStemList, []) #Using the principle of CSCP to get the optimal stems combinations
                            numCount = 0
                            for num in range(len(np.array(OptimalStemList))):
                                Rate = self.RateOfSterm(ResultStack[j], ResultStack[i], OptimalStemList[num], PreStructureList) # Calculate the Rate of PCRs
                                if(Rate > numCount):
                                    numCount = Rate
                            # When Rate is 1, restore all PCRs.
                            if(numCount >=1  and  ResultStack[j][2] > 1 and ResultStack[i][2] > 1):
                                TwoSuccess.append(ResultStack[j])
                                TwoSuccess.append(ResultStack[i])
                            if(numCount != 1 and (numCount >= (1+ ResultStack[j][2]/ResultStack[i][2])/2 or numCount >= (1+ ResultStack[i][2]/ResultStack[j][2])/2)):
                                TwoSuccess.append(ResultStack[j])
                                TwoSuccess.append(ResultStack[i])
        TwoSuccess = np.array(TwoSuccess)
        if(len(np.array(TwoSuccess)) != 0):
            TwoSuccess = TwoSuccess[np.lexsort(TwoSuccess[:, ::-1].T)]

        Successfully = []
        for i in range(len(TwoSuccess)): # Remove duplicate PCRs
            if(i == len(TwoSuccess)-1):
                Successfully.append(TwoSuccess[i])
            else:
                if(TwoSuccess[i][0] != TwoSuccess[i +1][0]):
                    Successfully.append(TwoSuccess[i])
        return Successfully
    def StermFreeCom(self, Resultstack, num): # Get the combination of PCRs.
        listnum = list(it.combinations(Resultstack, num))
        StermFreecom = []
        for i in range(len(np.array(listnum))):
            if (listnum[i][0][3] == 2 or listnum[i][-1][3] == 1):
                continue
            else:
                numOne = 0
                numTwo = 0
                TemOne = 0
                TemTwo = 0
                flag = False
                flagTwo = False
                for j in range(len(np.array(listnum[i]))):
                    if(listnum[i][j][2] != 2):
                        flagTwo = True
                    if(listnum[i][j][3] == 1):
                        if(TemTwo == 0):
                            TemOne += listnum[i][j][2]
                        else:
                            numOne += TemOne
                            numTwo += TemTwo
                            if(numOne < numTwo):
                                flag = True
                                break
                            else:
                                TemTwo = 0
                                TemOne = listnum[i][j][2]
                    if(listnum[i][j][3] == 2):
                        TemTwo += listnum[i][j][2]
                    if(j == len(np.array(listnum[i]))-1):
                        numOne += TemOne
                        numTwo += TemTwo
                        if(numOne < numTwo):
                            flag = True
                if(flag == False and numOne != 0 and numOne == numTwo and flagTwo == True):
                    StermFreecom.append(listnum[i])
        return StermFreecom
    def ListNumStermCreate(self, ResultStack, Matrix, PreStructureList):
        OptimalSterm = []
        for i in range(len(ResultStack)):
            if (ResultStack[i][3] == 1):
                continue
            else:
                IndexQ = ResultStack[i][1]
                IndexP = ResultStack[i][0]
                for j in range(i):
                    if (ResultStack[j][3] == 1):
                        IndexI = ResultStack[j][0]
                        IndexJ = ResultStack[j][1]  # +1
                        AllSterm, flag = self.MaxStem(IndexI, IndexJ, IndexP, IndexQ, Matrix)
                        SelectStemList = self.SelectStem(AllSterm, flag)
                        if (len(SelectStemList) != 0):
                            OptimalStemList = self.LayerStermCom(SelectStemList, [])
                            numCount = 0
                            for num in range(len(np.array(OptimalStemList))):
                                Rate = self.RateOfSterm(ResultStack[j], ResultStack[i], OptimalStemList[num],PreStructureList)
                                if (Rate > numCount):
                                    numCount = Rate
                            if ((numCount >= 1 and ResultStack[j][2] > 1 and ResultStack[i][2] > 1) or(numCount != 1 and (numCount >= (1 + ResultStack[j][2] / ResultStack[i][2]) / 2 or numCount >= (1 + ResultStack[i][2] / ResultStack[j][2]) / 2)) ):
                                OptimalSterm.append(OptimalStemList)
        return OptimalSterm
    def BasicCorrect(self, SuccessSterm, num, AllSterm, Matrix, PreStructureList):
        flag = False
        listnum = self.StermFreeCom(SuccessSterm, num)
        for i in range(len(np.array(listnum))):
            OptimalSterm = self.ListNumStermCreate(listnum[i], Matrix, PreStructureList)
            Sterm = []
            for numOne in range(len(np.array(OptimalSterm))):
                for j in range(len(np.array(OptimalSterm[numOne]))):
                    for num in range(len(np.array(OptimalSterm[numOne][j]))):
                        Sterm.append(OptimalSterm[numOne][j][num])
            if(len(np.array(Sterm)) != 0):
                SplitSterm = self.StermHeavyPair(Sterm)
                SplitSterm = np.array(SplitSterm)
                SplitSterm = SplitSterm[np.lexsort(-SplitSterm.T)]
                Combination = self.LayerStermCom(SplitSterm, [])
            else:
                Combination = []
            Prelength = 0
            for numTwo in range(len(np.array(listnum[i]))):
                if (listnum[i][numTwo][3] == 1):
                    Prelength += listnum[i][numTwo][2]
            StermLength = 0
            if(len(np.array(Combination)) != 0):
                for numThree in range(len(np.array(Combination[0]))):
                    StermLength += Combination[0][numThree][4]
            if(StermLength != Prelength or StermLength == 0):
                Combination = []
            else:
                flag = True
                for ii in range(len(np.array(Combination))):
                    for jj in range(len(np.array(Combination[ii]))):
                        AllSterm.append(Combination[ii][jj])
        return AllSterm, flag
    def RepairStack(self, ResultStack, Combination):
        RepairStack = []
        Count = 0
        for i in range(len(np.array(ResultStack))):
            flag = False
            for j in range(len(np.array(Combination))):
                if((Combination[j][0] <= ResultStack[i][0] and Combination[j][1] >= ResultStack[i][0]) or (Combination[j][0] <= ResultStack[i][1] and Combination[j][1] >= ResultStack[i][1]) or (Combination[j][2] <= ResultStack[i][0] and Combination[j][3] >= ResultStack[i][0]) or (Combination[j][2] <= ResultStack[i][1] and Combination[j][3] >= ResultStack[i][1])):
                    flag = True
                    break
                else:
                    continue
            if(flag == False):
                RepairStack.append(cy.deepcopy(ResultStack[i]))
                if(len(np.array(RepairStack)) == 1):
                    RepairStack[-1][4] = -1
                else:
                    if(RepairStack[-1][3] == RepairStack[-2][3]):
                        RepairStack[-1][4] = RepairStack[-1][0] - RepairStack[-2][1] -1
                    else:
                        RepairStack[-1][4] = -1
                RepairStack[-1][-1] = Count
                Count += 1
        return RepairStack
    def StermHeavyPair(self, MergeSterm):  # Remove repeated stem regions.
        SplitSterm = []
        for i in range(len(np.array(MergeSterm)) - 1):
            flag = False
            for j in range(i + 1, len(np.array(MergeSterm))):
                if (MergeSterm[i][0] == MergeSterm[j][0] and MergeSterm[i][1] == MergeSterm[j][1] and MergeSterm[i][2] == MergeSterm[j][2] and MergeSterm[i][3] == MergeSterm[j][3]):
                    flag = True
                    break
                else:
                    continue
            if (flag == False):
                SplitSterm.append(MergeSterm[i])
            else:
                continue
        SplitSterm.append(MergeSterm[-1])
        return SplitSterm

    def RemoveAllTwo(self, AllSterm, SuccessSterm):
        Sterm = []
        for i in range(len(np.array(AllSterm))):
            flagOne = False
            flagTwo = False
            for j in range(len(np.array(SuccessSterm))):
                if (AllSterm[i][0] >= SuccessSterm[j][0] and AllSterm[i][1] <= SuccessSterm[j][1] and SuccessSterm[j][2] != 2):
                    flagOne = True
                if (AllSterm[i][2] >= SuccessSterm[j][0] and AllSterm[i][3] <= SuccessSterm[j][1]):
                    if (SuccessSterm[j][2] != 2):
                        flagTwo = True
                    break
            if (flagOne or flagTwo):
                Sterm.append(AllSterm[i])
        return Sterm
    '''
    The CorrectSterm method is the FirstStep principle to get the first stems combinations.
    '''
    def CorrectSterm(self, SuccessSterm, Matrix, PreStructureList):
        TemSuccess = cy.deepcopy(SuccessSterm)
        Index = 2
        length = len(np.array(TemSuccess))
        AllSterm = []
        while(Index <= length and Index<=4):
            AllSterm, flag = self.BasicCorrect(TemSuccess, Index, AllSterm, Matrix, PreStructureList)
            if(flag):
                TemSuccess = self.RepairStack(TemSuccess, AllSterm)
                length = len(np.array(TemSuccess))
            Index += 1
        if(len(np.array(AllSterm)) != 0):
            AllSterm = self.StermHeavyPair(AllSterm)
            AllSterm = self.RemoveAllTwo(AllSterm, SuccessSterm) #
        AllSterm = np.array(AllSterm)
        if(len(np.array(AllSterm))!= 0):
            AllSterm = AllSterm[np.lexsort(-AllSterm.T)]
        Combination = self.LayerStermCom(AllSterm, [])
        return Combination

    # ******************************************Secondary Layer ********************************************************************
    def TmSecondaryMatrixStermCreate(self, ResultStack, Matrix, PreStructureList):
        AAllSterm = []
        for i in range(len(np.array(ResultStack))):
            if(ResultStack[i][3] == 1):
                continue
            else:
                IndexQ = ResultStack[i][1]
                IndexP = ResultStack[i][0]
                for j in range(i):
                    if (ResultStack[j][3] == 1):
                        IndexI = ResultStack[j][0]
                        IndexJ = ResultStack[j][1]  # +1
                        AllSterm, flag = self.MaxStem(IndexI, IndexJ, IndexP, IndexQ, Matrix)
                        SelectStemList = self.SelectStem(AllSterm,  flag)
                        StermLength = 0
                        if (len(SelectStemList) != 0):
                            OptimalStemList = self.LayerStermCom(SelectStemList, [])
                            for ii in range(len(np.array(OptimalStemList[0]))):
                                StermLength += OptimalStemList[0][ii][4]
                            numCount = 0
                            for num in range(len(np.array(OptimalStemList))):
                                if(ResultStack[j][2] > ResultStack[i][2]):
                                    lengthNum = ResultStack[j][3]
                                else:
                                    lengthNum = ResultStack[i][3]
                                SingleRate, _ = self.ScondarySingleRateOfSterm(ResultStack[j], ResultStack[i], OptimalStemList[num], PreStructureList, lengthNum)
                                if(SingleRate > numCount):
                                    numCount = SingleRate
                            if(numCount >= 1 and StermLength >= 5):
                                for iii in range(len(np.array(OptimalStemList))):
                                    for jjj in range(len(np.array(OptimalStemList[iii]))):
                                        AAllSterm.append(OptimalStemList[iii][jjj])
        AAllSterm = self.StermHeavyPair(AAllSterm)
        return AAllSterm
    def SecondaryMatrixStermCreate(self, ResultStack, Matrix, PreStructureList):
        TwoSuccess = []
        for i in range(len(np.array(ResultStack))):
            if(ResultStack[i][3] == 1):
                continue
            else:
                IndexQ = ResultStack[i][1]
                IndexP = ResultStack[i][0]
                for j in range(i):
                    if (ResultStack[j][3] == 1):
                        IndexI = ResultStack[j][0]
                        IndexJ = ResultStack[j][1]  # +1
                        AllSterm, flag = self.MaxStem(IndexI, IndexJ, IndexP, IndexQ, Matrix)
                        SelectStemList = self.SelectStem(AllSterm,  flag)
                        StermLength = 0
                        if (len(SelectStemList) != 0):
                            OptimalStemList = self.LayerStermCom(SelectStemList, [])
                            for ii in range(len(np.array(OptimalStemList[0]))):
                                StermLength += OptimalStemList[0][ii][4]
                            numCount = 0
                            for num in range(len(np.array(OptimalStemList))):
                                if(ResultStack[j][2] > ResultStack[i][2]):
                                    lengthNum = ResultStack[j][3]
                                else:
                                    lengthNum = ResultStack[i][3]
                                SingleRate, _ = self.ScondarySingleRateOfSterm(ResultStack[j], ResultStack[i], OptimalStemList[num], PreStructureList, lengthNum)
                                if(SingleRate > numCount):
                                    numCount = SingleRate
                            if(numCount >= 1):
                                for stermNum in range(len(np.array(OptimalStemList))):
                                    flagL = False
                                    for LNum in range(len(np.array(OptimalStemList[stermNum]))):
                                        if(OptimalStemList[stermNum][LNum][4] >= 4):
                                            flagL = True
                                            break
                                    if(flagL):
                                        for sNum in range(len(np.array(OptimalStemList[stermNum]))):
                                            TwoSuccess.append(OptimalStemList[stermNum][sNum])
        if(len(np.array(TwoSuccess))!= 0):
            Successfully = self.StermHeavyPair(TwoSuccess)
            Successfully = np.array(Successfully)
            Successfully =  Successfully[np.lexsort(-Successfully.T)]
        else:
            Successfully = []
        return Successfully
    def ThirdMatrixStermCreate(self, ResultStack, Matrix, PreStructureList):
        TwoSuccess = []  
        for i in range(len(np.array(ResultStack))):
            if(ResultStack[i][3] == 1):
                continue
            else:
                IndexQ = ResultStack[i][1]
                IndexP = ResultStack[i][0]
                for j in range(i):
                    if (ResultStack[j][3] == 1):
                        IndexI = ResultStack[j][0]
                        IndexJ = ResultStack[j][1]  # +1
                        AllSterm, flag = self.MaxStem(IndexI, IndexJ, IndexP, IndexQ, Matrix)
                        SelectStemList = self.SelectStem(AllSterm,  flag)
                        StermLength = 0
                        if (len(SelectStemList) != 0):
                            OptimalStemList = self.LayerStermCom(SelectStemList, [])
                            numCount = 0
                            for num in range(len(np.array(OptimalStemList))):
                                if(ResultStack[j][2] > ResultStack[i][2]):
                                    lengthNum = ResultStack[j][3]
                                else:
                                    lengthNum = ResultStack[i][3]
                                SingleRate, Rate = self.ScondarySingleRateOfSterm(ResultStack[j], ResultStack[i], OptimalStemList[num], PreStructureList, lengthNum)
                                if(Rate > numCount):
                                    numCount = Rate
                            FlagSterm = False
                            for stermNum in range(len(np.array(OptimalStemList))):
                                for sNum in range(len(np.array(OptimalStemList[stermNum]))):
                                    if(OptimalStemList[stermNum][sNum][4] >= 3):
                                        FlagSterm = True
                                        break
                            if(FlagSterm and numCount >= 0.4):
                                for stermNum in range(len(np.array(OptimalStemList))):
                                    for sNum in range(len(np.array(OptimalStemList[stermNum]))):
                                        TwoSuccess.append(OptimalStemList[stermNum][sNum])
                            if(FlagSterm == False and numCount >= 0.6):
                                for stermNum in range(len(np.array(OptimalStemList))):
                                    for sNum in range(len(np.array(OptimalStemList[stermNum]))):
                                        TwoSuccess.append(OptimalStemList[stermNum][sNum])
        if(len(np.array(TwoSuccess))!= 0):
            Successfully = self.StermHeavyPair(TwoSuccess)
            Successfully = np.array(Successfully)
            Successfully =  Successfully[np.lexsort(-Successfully.T)]
        else:
            Successfully = []
        return Successfully
    def LimitValue(self, ResultStack, LengthNum, Sterm):
        ForeValue = -1
        EaerValue = -1
        if(LengthNum == 1):
            for i in range(len(np.array(ResultStack))):
                if(Sterm[0] >= ResultStack[i][0] and Sterm[1] <= ResultStack[i][1]):
                    ForeValue = ResultStack[i][0]
                    EaerValue = ResultStack[i][1]
                    return ForeValue, EaerValue
        else:
            for i in range(len(np.array(ResultStack))):
                if(Sterm[2] >= ResultStack[i][0] and Sterm[3] <= ResultStack[i][1]):
                    ForeValue = ResultStack[i][0]
                    EaerValue = ResultStack[i][1]
                    return ForeValue, EaerValue
        return ForeValue, EaerValue
    def SecondaryExtendStem(self, Matrix, OptimalSterm,LengthNum, ResultStack):
        MedOptimal = []
        for numOne in range(len(np.array(OptimalSterm))):
            Tem = []
            I = int(OptimalSterm[numOne][0])
            J = int(OptimalSterm[numOne][1])
            P = int(OptimalSterm[numOne][2])
            Q = int(OptimalSterm[numOne][3])
            ForeValue, EaerValue = self.LimitValue(ResultStack, LengthNum,OptimalSterm[numOne])
            if(LengthNum == 1):
                while(I > 0 and Q < len(np.array(Matrix))-1 and I >= ForeValue):
                    if(Matrix[I-1][Q + 1] == 1):
                        I -= 1
                        Q += 1
                    else:
                        break
                while(J < len(np.array(Matrix))-1 and P > 0 and J<= EaerValue):
                    if (Matrix[J+1][P-1] == 1):
                        J += 1
                        P -= 1
                    else:
                        break
            else:
                while (I > 0 and Q < len(np.array(Matrix)) - 1 and Q <= EaerValue):
                    if (Matrix[I - 1][Q + 1] == 1):
                        I -= 1
                        Q += 1
                    else:
                        break
                while (J < len(np.array(Matrix)) - 1 and P > 0 and P >= ForeValue):
                    if (Matrix[J + 1][P - 1] == 1):
                        J += 1
                        P -= 1
                    else:
                        break
            Tem.append(I)
            Tem.append(J)
            Tem.append(P)
            Tem.append(Q)
            Tem.append(J-I + 1)
            MedOptimal.append(Tem)
        OptimalSterm = MedOptimal
        return OptimalSterm
    def ScondarySingleRateOfSterm(self, LocalOne, LocalTwo, OptimalStemList, PreStructure,LengthNum):
        RateOne = 0
        RateTwo = 0
        for i in range(len(np.array(OptimalStemList))):
            for j in range(int(OptimalStemList[i][0]), int(OptimalStemList[i][1]) + 1):
                if (j >= LocalOne[0] and j <= LocalOne[1] and PreStructure[j] == 1):
                    RateOne += 1
        for i in range(len(np.array(OptimalStemList))):
            for j in range(int(OptimalStemList[i][2]), int(OptimalStemList[i][3]) + 1):
                if (j >= LocalTwo[0] and j <= LocalTwo[1] and PreStructure[j] == 2):
                    RateTwo += 1
        if(LengthNum == 1):
            return RateTwo / LocalTwo[2], (RateTwo / LocalTwo[2] + RateOne / LocalOne[2]) / 2
        else:
            return RateOne / LocalOne[2], (RateTwo / LocalTwo[2] + RateOne / LocalOne[2]) / 2
    def SingleRateOfSterm(self, LocalOne, LocalTwo, OptimalStemList, PreStructure,LengthNum):
        RateOne = 0
        RateTwo = 0
        for i in range(len(np.array(OptimalStemList))):
            for j in range(int(OptimalStemList[i][0]), int(OptimalStemList[i][1]) + 1):
                if (j >= LocalOne[0] and j <= LocalOne[1] and PreStructure[j] == 1):
                    RateOne += 1
        for i in range(len(np.array(OptimalStemList))):
            for j in range(int(OptimalStemList[i][2]), int(OptimalStemList[i][3]) + 1):
                if (j >= LocalTwo[0] and j <= LocalTwo[1] and PreStructure[j] == 2):
                    RateTwo += 1
        if(LengthNum == 1):
            return RateOne / LocalOne[2], (RateTwo / LocalTwo[2] + RateOne / LocalOne[2]) / 2
        else:
            return RateTwo / LocalTwo[2], (RateTwo / LocalTwo[2] + RateOne / LocalOne[2]) / 2
    def SecondaryListNumStermCreate(self, ResultStack, Matrix, PreStructureList, LengthNum):
        OptimalSterm = []
        for i in range(len(ResultStack)):
            if (ResultStack[i][3] == 1):
                continue
            else:
                IndexQ = ResultStack[i][1]
                IndexP = ResultStack[i][0]
                for j in range(i):
                    if (ResultStack[j][3] == 1):
                        IndexI = ResultStack[j][0]
                        IndexJ = ResultStack[j][1]
                        AllSterm, flag = self.MaxStem(IndexI, IndexJ, IndexP, IndexQ, Matrix)
                        SelectStemList = self.SelectStem(AllSterm, flag)
                        SelectStemList = self.SecondaryExtendStem(Matrix, SelectStemList,LengthNum, ResultStack)
                        if(len(np.array(SelectStemList)) != 0):
                            SelectStemList = self.StermHeavyPair(SelectStemList)
                        if (len(SelectStemList) != 0):
                            OptimalStemList = self.LayerStermCom(SelectStemList, [])
                            numCount = 0
                            SingleCount = 0
                            for num in range(len(np.array(OptimalStemList))):
                                SingleRate, Rate = self.SingleRateOfSterm(ResultStack[j], ResultStack[i], OptimalStemList[num],PreStructureList, LengthNum)
                                if (Rate > numCount):
                                    numCount = Rate
                                if(SingleRate > SingleCount):
                                    SingleCount = SingleRate
                            if ((SingleCount >= 1) or(numCount != 1 and (numCount >= (1 + ResultStack[j][2] / ResultStack[i][2]) / 2 or numCount >= (1 + ResultStack[i][2] / ResultStack[j][2]) / 2)) ):
                                OptimalSterm.append(OptimalStemList)
        return OptimalSterm
    def SecondaryStermFreeCom(self, Resultstack, num):
        listnum = list(it.combinations(Resultstack, num))
        StermFreeCom = []
        for i in range(len(np.array(listnum))):
            if (listnum[i][0][3] == 2 or listnum[i][-1][3] == 1):
                continue
            else:
                numOne = 0
                numTwo = 0
                flagTwo = False
                for j in range(len(np.array(listnum[i]))):
                    if (listnum[i][j][2] != 2):
                        flagTwo = True
                    if (listnum[i][j][3] == 1):
                        numOne += listnum[i][j][2]
                    if (listnum[i][j][3] == 2):
                        numTwo += listnum[i][j][2]
                if (numOne != 0 and numTwo != 0 and numOne != numTwo and flagTwo == True):
                    StermFreeCom.append(listnum[i])
        return StermFreeCom
    def SecondaryBasicCorrect(self, SuccessSterm, num, AllSterm, Matrix, PreStructureList):
        flag = False
        listnum = self.SecondaryStermFreeCom(SuccessSterm, num)
        for i in range(len(np.array(listnum))):
            One = 0
            Two = 0
            for NumTwo in range(len(np.array(listnum[i]))):
                if (listnum[i][NumTwo][3] == 1):
                    One += listnum[i][NumTwo][2]
            for NumTwoo in range(len(np.array(listnum[i]))):
                if (listnum[i][NumTwoo][3] == 2):
                    Two += listnum[i][NumTwoo][2]
            if(One > Two):
                LengthNum = 1
                GoalLength = One
            else:
                GoalLength = Two
                LengthNum = 2
            OptimalSterm = self.SecondaryListNumStermCreate(listnum[i], Matrix, PreStructureList, LengthNum)
            Sterm = []
            for numOne in range(len(np.array(OptimalSterm))):
                for j in range(len(np.array(OptimalSterm[numOne]))):
                    for num in range(len(np.array(OptimalSterm[numOne][j]))):
                        Sterm.append(OptimalSterm[numOne][j][num])
            if(len(np.array(Sterm)) != 0):
                SplitSterm = self.StermHeavyPair(Sterm)
                SplitSterm = np.array(SplitSterm)
                SplitSterm = SplitSterm[np.lexsort(-SplitSterm.T)]
                Combination = self.LayerStermCom(SplitSterm, [])
            else:
                Combination = []

            Prelength = 0
            if(LengthNum == 1):
                for numTwo in range(len(np.array(listnum[i]))):
                    if (listnum[i][numTwo][3] == 1):
                        Prelength += listnum[i][numTwo][2]
            else:
                for numTwoo in range(len(np.array(listnum[i]))):
                    if (listnum[i][numTwoo][3] == 2):
                        Prelength += listnum[i][numTwoo][2]
            StermLength = 0
            if(len(np.array(Combination)) != 0):
                for numThree in range(len(np.array(Combination[0]))):
                    StermLength += Combination[0][numThree][4]
            if(StermLength != Prelength or StermLength == 0 or GoalLength < 4):
                Combination = []
            else:
                flag = True
                for ii in range(len(np.array(Combination))):
                    for jj in range(len(np.array(Combination[ii]))):
                        AllSterm.append(Combination[ii][jj])
        return AllSterm, flag
    def SecondaryCorrectSterm(self, SuccessSterm, Matrix, PreStructureList, CombinationFist):
        TemSuccess = cy.deepcopy(SuccessSterm)
        Index = 2
        length = len(np.array(TemSuccess))
        AllSterm = []
        while(Index <= 4):
            AllSterm, flag = self.SecondaryBasicCorrect(TemSuccess, Index, AllSterm, Matrix, PreStructureList)
            if(flag):
                TemSuccess = self.RepairStack(TemSuccess, AllSterm)
                length = len(np.array(TemSuccess))
            Index += 1
        if(len(np.array(AllSterm)) != 0):
            AllSterm = self.StermHeavyPair(AllSterm)
            AllSterm = self.RemoveAllTwo(AllSterm, SuccessSterm)
        CombinationSecondary = self.LayerStermCom(AllSterm, CombinationFist)
        return CombinationSecondary
    def ValueOfPPVSen(self, PriList, OptimalStemList):
        MedList = []
        ValueList = []
        AllPairs = 0
        TruePositive = 0
        FalsePositive = 0
        TrueNegative = 0
        FalseNegative = 0
        AllPredicPairs = 0
        ValurArray = np.zeros(shape=[len(PriList)])
        for i in range(len(OptimalStemList)):
            IndexI = OptimalStemList[i][0]
            IndexJ = OptimalStemList[i][1]
            IndexP = OptimalStemList[i][2]
            IndexQ = OptimalStemList[i][3]
            for j in range(int(IndexI), int(IndexJ + 1)):
                ValurArray[j] = OptimalStemList[i][3] - (j - IndexI) + 1
            count = 0
            while (IndexQ >= IndexP):
                ValurArray[int(IndexQ)] = OptimalStemList[i][0] + count + 1
                count += 1
                IndexQ -= 1
        for i in range(len(PriList)):
            MedList.append(PriList[i][0])
            MedList.append(PriList[i][1])
            MedList.append(PriList[i][2])
            MedList.append(ValurArray[i])
            ValueList.append(MedList)
            MedList = []
        ValueList = np.array(ValueList)
        for i in range(len(ValueList)):
            if (ValueList[i][2] != 0 and ValueList[i][2] > ValueList[i][0]):
                AllPairs += 1
            if (ValueList[i][3] != 0 and ValueList[i][3] > ValueList[i][0]):
                AllPredicPairs += 1
                if (ValueList[i][2] == ValueList[i][3]):
                    TruePositive += 1
                else:
                    FalsePositive += 1
            else:
                if (ValueList[i][2] == 0):
                    TrueNegative += 1
                else:
                    FalseNegative += 1

        if (AllPredicPairs != 0 and AllPairs != 0):
            PPV = TruePositive / AllPredicPairs
            Sensitive = TruePositive / AllPairs
            if(PPV+Sensitive != 0):
                Fscore = 2*(PPV*Sensitive/(PPV+Sensitive))
            else:
                Fscore = 0
        else:
            PPV = 0
            Sensitive = 0
            Fscore = 0
            # 计算 MCC

        denominator = np.sqrt(
            (TruePositive + FalsePositive) * (TruePositive + FalseNegative) * (TrueNegative + FalsePositive) * (
                        TrueNegative + FalseNegative))
        if denominator != 0:
            MCC = (TruePositive * TrueNegative - FalsePositive * FalseNegative) / denominator
        else:
            MCC = 0
        return PPV, Sensitive, Fscore, MCC
    # ******************************************Secondary Layer ********************************************************************
    def SynCollection(self, PositionOne, ResultStack):
        CollectionSterm = []
        Index = PositionOne
        while (Index > 0):
            if (ResultStack[Index][4] <= 3 and ResultStack[Index][4] != -1):
                CollectionSterm.append(ResultStack[Index - 1])
            else:
                break
            Index -= 1
        CollectionSterm.reverse()
        CollectionSterm.append(ResultStack[PositionOne])
        for i in range(PositionOne + 1, len(np.array(ResultStack))):
            if (ResultStack[i][4] <= 3 and ResultStack[i][4] != -1):
                CollectionSterm.append(ResultStack[i])
            else:
                break
        return CollectionSterm
    def CollectionDoubleSterm(self, PositionOne, PositionTwo, ResultStack):
        CollectionSterm = []
        Med = []
        Index = PositionOne
        while (Index > 0):
            if (ResultStack[Index][4] <= 3 and ResultStack[Index][4] != -1):
                CollectionSterm.append(ResultStack[Index - 1])
            else:
                break
            Index -= 1
        CollectionSterm.reverse()
        CollectionSterm.append(ResultStack[PositionOne])
        for i in range(PositionOne + 1, PositionTwo):
            if (ResultStack[i][4] <= 3 and ResultStack[i][4] != -1):
                CollectionSterm.append(ResultStack[i])
            else:
                break
        Index = PositionTwo
        while (Index > PositionOne):
            if (ResultStack[Index][4] <= 3 and ResultStack[Index][4] != -1):
                Med.append(ResultStack[Index - 1])
            else:
                break
            Index -= 1
        Med.reverse()
        for i in range(len(np.array(Med))):
            CollectionSterm.append(Med[i])
        CollectionSterm.append(ResultStack[PositionTwo])
        for i in range(PositionTwo + 1, len(np.array(ResultStack))):
            if (ResultStack[i][4] <= 3 and ResultStack[i][4] != -1):
                CollectionSterm.append(ResultStack[i])
            else:
                break
        return CollectionSterm
    def SynSterm(self, CollectionSterm):
        MedSynLargeSterm = []
        SynLargeSterm = []
        Index = len(np.array(CollectionSterm))-1
        while(Index > 0):
            TemList = []
            TemList.append(CollectionSterm[Index])
            IndexNum = Index - 1
            while(IndexNum >= 0):
                if(CollectionSterm[Index][4] <= 3 and CollectionSterm[Index][4] != -1 and CollectionSterm[Index][7] -1 == CollectionSterm[IndexNum][7]):
                    TemList.append(CollectionSterm[IndexNum])
                    Index -= 1
                    IndexNum -= 1
                else:
                    TemList.reverse()
                    MedSynLargeSterm.append(TemList)
                    if(IndexNum==0):
                        TemList = []
                        TemList.append(CollectionSterm[IndexNum])
                        MedSynLargeSterm.append(TemList)
                    break
                if(IndexNum < 0):
                    TemList.reverse()
                    MedSynLargeSterm.append(TemList)
                    break
            Index -= 1
        MedSynLargeSterm.reverse()
        for i in range(len(np.array(MedSynLargeSterm))):
            TemSyn = []
            count = 0
            for j in range(len(np.array(MedSynLargeSterm[i]))):
                count += MedSynLargeSterm[i][j][2]
                if(j == 0):
                    TemSyn.append(MedSynLargeSterm[i][j][0])
                if(j == len(np.array(MedSynLargeSterm[i]))-1):
                    TemSyn.append(MedSynLargeSterm[i][j][1])
                    TemSyn.append(count)
                    TemSyn.append(MedSynLargeSterm[i][j][3])
                    TemSyn.append(0)
                    TemSyn.append(0)
                    TemSyn.append(0)
                    TemSyn.append(0)
            SynLargeSterm.append(TemSyn)
        return SynLargeSterm
    def SecMatrixStermCreate(self, ResultStack, Matrix, PreStructureList):
        MatrixSterm = np.zeros(shape=[len(np.array(ResultStack)), len(np.array(ResultStack))])
        PairedSterm = np.zeros(shape=[len(np.array(ResultStack)), len(np.array(ResultStack))])
        Paired = []
        PairIndex = 0
        for i in range(len(MatrixSterm)):
            if(ResultStack[i][3] == 1):
                continue
            else:
                IndexQ = ResultStack[i][1]
                IndexP = ResultStack[i][0]
                for j in range(i):
                    if (ResultStack[j][3] == 1):
                        IndexI = ResultStack[j][0]
                        IndexJ = ResultStack[j][1]
                        AllSterm, flag = self.MaxStem(IndexI, IndexJ, IndexP, IndexQ, Matrix)
                        SelectStemList = self.SelectStem(AllSterm,  flag)
                        if (len(SelectStemList) != 0):
                            OptimalStemList = self.LayerStermCom(SelectStemList, [])
                            numCount = 0
                            for num in range(len(np.array(OptimalStemList))):
                                Rate = self.RateOfSterm(ResultStack[j], ResultStack[i], OptimalStemList[num], PreStructureList)
                                if(Rate > numCount):
                                    numCount = Rate
                            MatrixSterm[i][j] = numCount
                            Paired.append(OptimalStemList)
                            PairedSterm[i][j] = PairIndex
                            PairIndex += 1
                        else:
                            MatrixSterm[i][j] = -1
        return MatrixSterm, PairedSterm, Paired
    def IntellSelectSterm(self, SynSterm, Matrix, OptimalSterm, ExtendOptimalSterm, Prestructure):
        Mat = cy.deepcopy(Matrix)
        for num in range(len(np.array(ExtendOptimalSterm))):
            for CountI in range(ExtendOptimalSterm[num][0], ExtendOptimalSterm[num][1] + 1):
                for n in range(len(np.array(Matrix))):
                    Mat[CountI][n] = 0
                    Mat[n][CountI] = 0
            for CountP in range(ExtendOptimalSterm[num][2], ExtendOptimalSterm[num][3] + 1):
                for nn in range(len(np.array(Matrix))):
                    Mat[CountP][nn] = 0
                    Mat[nn][CountP] = 0

        for Num in range(len(np.array(SynSterm))):
            if(SynSterm[Num][0] == 0):
                continue
            else:
                if(Num == 0):
                    for NumOne in range(0, SynSterm[Num][0]):
                        for nn in range(len(np.array(Matrix))):
                            Mat[NumOne][nn] = 0
                            Mat[nn][NumOne] = 0
                elif(Num == len(np.array(SynSterm))-1):
                    if(SynSterm[Num][1] < len(Prestructure)-1):
                        for NumTwo in range(SynSterm[Num-1][1]+1, SynSterm[Num][0]):
                            for nn in range(len(np.array(Matrix))):
                                Mat[NumTwo][nn] = 0
                                Mat[nn][NumTwo] = 0
                        for NumThree in range(SynSterm[Num][1]+1, len(Prestructure)):
                            for nn in range(len(np.array(Matrix))):
                                Mat[NumThree][nn] = 0
                                Mat[nn][NumThree] = 0
                    else:
                        for NumTwo in range(SynSterm[Num-1][1]+1, SynSterm[Num][0]):
                            for nn in range(len(np.array(Matrix))):
                                Mat[NumTwo][nn] = 0
                                Mat[nn][NumTwo] = 0
                else:
                    for NumFour in range(SynSterm[Num-1][1]+1, SynSterm[Num][0]):
                        for nn in range(len(np.array(Matrix))):
                            Mat[NumFour][nn] = 0
                            Mat[nn][NumFour] = 0

        AllSterm, flag = self.MaxStem(0, len(np.array(Prestructure))-1, 0, len(np.array(Prestructure))-1, Mat)
        TList = []
        for i in range(len(np.array(AllSterm))):
            ForeNum = -1
            ErarNum = -1
            for j in range(len(np.array(SynSterm))):
                if(AllSterm[i][0] >= SynSterm[j][0] and AllSterm[i][1] <= SynSterm[j][1]):
                    ForeNum = j
                if(AllSterm[i][2] >= SynSterm[j][0] and AllSterm[i][3] <= SynSterm[j][1]):
                    ErarNum = j
            if(SynSterm[ForeNum][3] == 1 and SynSterm[ErarNum][3] == 2 ):
                TList.append(AllSterm[i])
            else:
                continue
        SelectStemList = self.SelectStem(TList, flag)
        TemList =[]
        for i in range(len(np.array(SelectStemList))):
            flag = False
            for j in range(len(np.array(OptimalSterm))):
                if(SelectStemList[i][3] < OptimalSterm[j][0] or OptimalSterm[j][3] < SelectStemList[i][0] or (SelectStemList[i][1]<OptimalSterm[j][0] and OptimalSterm[j][3]<SelectStemList[i][2]) or (OptimalSterm[j][1]<SelectStemList[i][0] and SelectStemList[i][3]<OptimalSterm[j][2])):
                    continue
                else:
                    flag = True
                    break
            if(flag == False):
                TemList.append(SelectStemList[i])
            else:
                continue
        NewOptimalSterm = []
        if(len(np.array(TemList)) == 0):
            NewOptimalSterm = [ExtendOptimalSterm]
        else:
            Combination = [OptimalSterm]
            OptimalSterm = self.LayerStermCom(TemList, Combination)
            OptimalSterm = np.array(OptimalSterm)
            for i in range(len(np.array(OptimalSterm))):
                Tem = self.IntellExtendStem(Matrix, OptimalSterm[i], Prestructure, 1, 2)
                NewOptimalSterm.append(Tem)
        return NewOptimalSterm
    def IntellExtendStem(self, Matrix, OptimalSterm, Prestructure, PreNum, ErarNum): # Extend the prediction results.
        Mat = cy.deepcopy(Matrix)
        for ii in range(len(np.array(OptimalSterm))):
            for nn in range(int(OptimalSterm[ii][0]), int(OptimalSterm[ii][1]+1)):
                for mm in range(len(np.array(Mat))):
                    Mat[nn][mm] = 0
                    Mat[mm][nn] = 0
            for pp in range(int(OptimalSterm[ii][2]), int(OptimalSterm[ii][3]+1)):
                for qq in range(len(np.array(Mat))):
                    Mat[qq][pp] = 0
                    Mat[pp][qq] = 0

        MedOptimal = []
        for numOne in range(len(np.array(OptimalSterm))):
            Tem = []
            I = int(OptimalSterm[numOne][0])
            J = int(OptimalSterm[numOne][1])
            P = int(OptimalSterm[numOne][2])
            Q = int(OptimalSterm[numOne][3])
            while(I > 0 and Q < len(np.array(Mat))-1):
                if(Mat[I-1][Q + 1] == 1):
                    I -= 1
                    Q += 1
                    for numTwo in range(len(np.array(Mat))):
                        Mat[I][numTwo] = -1
                        Mat[numTwo][Q] = -1
                elif(Mat[I-1][Q + 1] == -1 and Matrix[I-1][Q+1] == 1):
                    Index = 0
                    Falg = False
                    while(Index < len(np.array(MedOptimal))):
                        if(I-1 == MedOptimal[Index][1]):
                            if(Prestructure[Q+1]==ErarNum and  Prestructure[MedOptimal[Index][2]] != ErarNum):
                                I -= 1
                                Q += 1
                                MedOptimal[Index][1] -= 1
                                MedOptimal[Index][2] += 1
                                MedOptimal[Index][4] = MedOptimal[Index][1] - MedOptimal[Index][0] + 1
                                for numThree in range(len(np.array(Mat))):
                                    Mat[numThree][Q] = -1
                                    Mat[numThree][MedOptimal[Index][2]-1] = Matrix[numThree][MedOptimal[Index][2]-1]
                            else:
                                Falg = True
                                break
                        elif(Q+1 == MedOptimal[Index][2]):
                            if(Prestructure[I-1] == PreNum and Prestructure[MedOptimal[Index][1]] != PreNum):
                                I -= 1
                                Q += 1
                                MedOptimal[Index][1] -= 1
                                MedOptimal[Index][2] += 1
                                MedOptimal[Index][4] = MedOptimal[Index][1] - MedOptimal[Index][0] + 1
                                for numFour in range(len(np.array(Matrix))):
                                    Mat[I][numFour] = -1
                                    Mat[MedOptimal[Index][1]+1][numFour] = Matrix[MedOptimal[Index][1] + 1][numFour]
                            else:
                                Falg = True
                                break
                        else:
                            if(Index == len(np.array(MedOptimal)) - 1):
                                Falg = True
                                break
                            Index += 1
                    if(Falg):
                        break

                else:
                    break
            while(J < len(np.array(Mat))-1 and P > 0):
                if (Mat[J+1][P-1] == 1):
                    J += 1
                    P -= 1
                    for numFive in range(len(np.array(Mat))):
                        Mat[J][numFive] = -1
                        Mat[numFive][P] = -1
                elif(Mat[J+1][P - 1] == -1 and Matrix[J+1][P - 1] == 1):
                    Index = 0
                    Falg = False
                    while (Index < len(np.array(MedOptimal))):
                        if (J + 1 == MedOptimal[Index][0]):
                            if (Prestructure[P - 1] == ErarNum and  Prestructure[MedOptimal[Index][3]] != ErarNum):
                                J += 1
                                P -= 1
                                MedOptimal[Index][0] += 1
                                MedOptimal[Index][3] -= 1
                                MedOptimal[Index][4] = MedOptimal[Index][1] - MedOptimal[Index][0] + 1
                                for numSix in range(len(np.array(Mat))):
                                    Mat[numSix][P] = -1
                                    Mat[numSix][MedOptimal[Index][3] + 1] = Matrix[numSix][MedOptimal[Index][3] + 1]
                            else:
                                Falg = True
                                break
                        elif(P - 1 == MedOptimal[Index][3]):
                            if (Prestructure[J + 1] == PreNum and  Prestructure[MedOptimal[Index][0]] != PreNum):
                                J += 1
                                P -= 1
                                MedOptimal[Index][0] += 1
                                MedOptimal[Index][3] -= 1
                                MedOptimal[Index][4] = MedOptimal[Index][1] - MedOptimal[Index][0] + 1
                                for numSeven in range(len(np.array(Mat))):
                                    Mat[J][numSeven] = -1
                                    Mat[MedOptimal[Index][0] - 1][numSeven] = Matrix[MedOptimal[Index][0] - 1][numSeven]
                            else:
                                Falg = True
                                break
                        else:
                            if(Index == len(np.array(MedOptimal)) - 1):
                                Falg = True
                                break
                            Index += 1
                    if (Falg):
                        break

                else:
                    break
            Tem.append(I)
            Tem.append(J)
            Tem.append(P)
            Tem.append(Q)
            Tem.append(J-I + 1)
            MedOptimal.append(Tem)
        OptimalSterm = MedOptimal
        return OptimalSterm
    def DoubleSingleSmallSterm(self, CollecSterm,PreStructure, Matrix, CombinationOne):
        Sterm = []
        MatrixSterm, PairedSterm, Paired = self.SecMatrixStermCreate(CollecSterm, Matrix, PreStructure)
        for i in range(len(np.array(CollecSterm))):
            if(CollecSterm[i][3] == 1):
                continue
            else:
                Index = i -1
                while(Index >= 0):
                    if(CollecSterm[Index][3] == 2):
                        Index -= 1
                        continue
                    else:
                        if(MatrixSterm[i][Index] >= (1 + CollecSterm[i][2]/CollecSterm[Index][2])/2 or MatrixSterm[i][Index] >= (1 + CollecSterm[Index][2]/CollecSterm[i][2])/2 or (CollecSterm[i][2]>=4 and CollecSterm[Index][2]>=4 and MatrixSterm[i][Index]>0.8)):
                            CombinationSterm = Paired[int(PairedSterm[i][Index])]
                            for j in range(len(np.array(CombinationSterm))):
                                for num in range(len(np.array(CombinationSterm[j]))):
                                    Sterm.append(CombinationSterm[j][num])
                    Index -= 1
        if(len(np.array(Sterm)) == 0):
            Combination = CombinationOne
        else:
            Sterm = self.StermHeavyPair(Sterm)
            Sterm = np.array(Sterm)
            SelectStemListSort = Sterm[np.lexsort(-Sterm.T)]
            Combination = self.LayerStermCom(SelectStemListSort, CombinationOne)
        IntellSelect = []
        IntellSelectOptimal = []
        SynLargeSterm = self.SynSterm(CollecSterm)
        if(len(np.array(Combination)) == 0):
            MedIntellSelect = self.IntellSelectSterm(SynLargeSterm, Matrix, np.array([]), [], PreStructure)
            for j in range(len(np.array(MedIntellSelect))):
                IntellSelect.append(MedIntellSelect[j])
        else:
            for num in range(len(np.array(Combination))):
                Med = self.IntellExtendStem(Matrix, Combination[num], PreStructure, 1, 2)
                MedIntellSelect = self.IntellSelectSterm(SynLargeSterm, Matrix, Combination[num], Med, PreStructure)
                for j in range(len(np.array(MedIntellSelect))):
                    IntellSelect.append(MedIntellSelect[j])
        numList = []
        for i in range(len(np.array(IntellSelect))):
            num = 0
            for j in range(len(np.array(IntellSelect[i]))):
                num += IntellSelect[i][j][4]
            numList.append(num)
        if(len(np.array(numList)) != 0):
            largeNum = numList[0]
            for i in range(len(np.array(numList))):
                if (numList[i] > largeNum):
                    largeNum = numList[i]
            for i in range(len(np.array(numList))):
                if (numList[i] == largeNum):
                    IntellSelectOptimal.append(IntellSelect[i])
        return IntellSelectOptimal
    def Rate(self, CollecSterm, Combination):
        OneSterm = []
        OneLength = 0
        TwoSterm = []
        TwoLength = 0
        numOne = 0
        numTwo = 0
        for i in range(len(np.array(CollecSterm))):
            if(CollecSterm[i][3] == 1 or CollecSterm[i][3] == 3 or CollecSterm[i][3] == 5):
                OneSterm.append(CollecSterm[i])
                OneLength += CollecSterm[i][2]
            else:
                TwoSterm.append(CollecSterm[i])
                TwoLength += CollecSterm[i][2]
        for i in range(len(np.array(Combination))):
            for p in range(Combination[i][0], Combination[i][1] + 1):
                flagOne = False
                for q in range(len(np.array(OneSterm))):
                    if(p<OneSterm[q][0] or p>OneSterm[q][1]):
                        continue
                    else:
                        flagOne = True
                        break
                if(flagOne):
                    numOne += 1
            for pp in range(Combination[i][2], Combination[i][3] + 1):
                flagTwo = False
                for qq in range(len(np.array(TwoSterm))):
                    if (pp < TwoSterm[qq][0] or pp > TwoSterm[qq][1]):
                        continue
                    else:
                        flagTwo = True
                        break
                if(flagTwo):
                    numTwo += 1
        rateOne = numOne/OneLength
        rateTwo = numTwo/TwoLength
        return rateOne, rateTwo
    def CreateSynMatrix(self, ResultStack, Matrix, PreStructure):
        Med = []
        Index = 0
        while(Index < len(np.array(ResultStack))):
            CollectionSterm = self.SynCollection(Index, ResultStack)
            pos = -1
            num = -1
            for i in range(len(np.array(CollectionSterm))):
                if(CollectionSterm[i][2] > num):
                    num = CollectionSterm[i][2]
                    pos = i
            Med.append(CollectionSterm[pos])
            Index = CollectionSterm[-1][7] + 1
        SynStermMatrixNone = np.zeros(shape=[len(np.array(Med)), len(np.array(Med))])
        countMatrix = np.zeros(shape=[len(np.array(Med)), len(np.array(Med))])
        CombinationList = []
        count = -1

        for i in range(len(np.array(Med))):
            if (Med[i][3] == 1):
                continue
            else:
                PositionTwo = Med[i][7]
                for j in range(i):
                    if (Med[j][3] == 1):
                        PositionOne = Med[j][7]
                        CollecSterm = self.CollectionDoubleSterm(PositionOne, PositionTwo, ResultStack)
                        CombinationNone = self.DoubleSingleSmallSterm(CollecSterm,PreStructure, Matrix, [])
                        count += 1
                        CombinationList.append(CombinationNone)
                        countMatrix[i][j] = count
                        CorrectRate = -1
                        CollecSterm = self.CollectionDoubleSterm(PositionOne, PositionTwo, ResultStack)
                        if (len(np.array(CombinationNone)) == 1):
                            rateOne, rateTwo = self.Rate(CollecSterm, CombinationNone[0])
                            SynStermMatrixNone[i][j] = (rateOne + rateTwo) / 2
                        elif (len(np.array(CombinationNone)) == 0):
                            SynStermMatrixNone[i][j] = -1
                        else:
                            for num in range(len(np.array(CombinationNone))):
                                rateOne, rateTwo = self.Rate(CollecSterm, CombinationNone[num])
                                if ((rateOne + rateTwo) / 2 > CorrectRate):
                                    CorrectRate = (rateOne + rateTwo) / 2
                            SynStermMatrixNone[i][j] = CorrectRate
        return SynStermMatrixNone, Med, countMatrix, CombinationList
    def CombinationStermCollection(self, Combination):
        SplitList = []
        for i in range(len(np.array(Combination))):
            for j in range(len(np.array(Combination[i]))):
                SplitList.append(Combination[i][j])
        return SplitList
    def FinalResult(self, SynStermMatrixNone, Med, SynSterm, countMatrix, CombinationList, Combination):
        CorrectSterm = []
        for i in range(len(np.array(SynStermMatrixNone))):
            if (Med[i][3] == 1 or  Med[i][5] == 1):
                continue
            else:
                j = i - 1
                while (j >= 0):
                    flag = False
                    if (SynStermMatrixNone[i][j] >= 0.9 and Med[j][5] == 0 and Med[i][2] >= 4 and Med[j][2] >= 4):
                        Med[i][5] = 1
                        Med[j][5] = 1
                        flag = True
                        Split = self.CombinationStermCollection(CombinationList[int(countMatrix[i][j])])
                        for num in range(len(np.array(Split))):
                            CorrectSterm.append(Split[num])
                    if (flag):
                        break
                    j -= 1
        for i in range(len(np.array(SynStermMatrixNone))):
            if (Med[i][3] == 1 or Med[i][5] == 1):
                continue
            else:
                j = i - 1
                while (j >= 0):
                    if((Med[j][3] == 2 or Med[j][3] == 4 or Med[j][3] == 6) and Med[j][2] > 1 and Med[j][5] == 0):
                        break
                    if((Med[j][3] == 1 or Med[j][3] == 3 or Med[j][3] == 5) and Med[j][2] > 1 and SynStermMatrixNone[i][j]<0.81 and Med[j][5] == 0):
                        break
                    if ((SynStermMatrixNone[i][j] >= 0.81 and Med[j][5] == 0) or(SynStermMatrixNone[i][j] >= 0.75 and Med[j][5] == 0 and CombinationList[int(countMatrix[i][j])][0][0][4] >=5)):
                        Med[i][5] = 1
                        Med[j][5] = 1
                        Split = self.CombinationStermCollection(CombinationList[int(countMatrix[i][j])])
                        for num in range(len(np.array(Split))):
                            CorrectSterm.append(Split[num])
                        break
                    j -= 1
        if (len(np.array(CorrectSterm)) == 0):
            Combination = Combination
        else:
            CorrectSterm = self.StermHeavyPair(CorrectSterm)
            CorrectSterm = np.array(CorrectSterm)
            SelectStemListSort = CorrectSterm[np.lexsort(-CorrectSterm.T)]
            Combination = self.LayerStermCom(SelectStemListSort, Combination)
        RemainMed = []
        for i in range(len(np.array(Med))):
            if (Med[i][5] == 0):
                RemainMed.append(Med[i])
        return Combination, RemainMed
    def Collection(self, PositionOne, PositionTwo, ResultStack):
        CollectionSterm = []
        Index = PositionOne
        while(Index > 0):
            if(ResultStack[Index][4]<= 3 and ResultStack[Index][4] != -1 and ResultStack[Index-1][5] == 0):
                CollectionSterm.append(ResultStack[Index-1])
            else:
                break
            Index -= 1
        CollectionSterm.reverse()
        for i in range(PositionOne, PositionTwo+1):
            CollectionSterm.append(ResultStack[i])
        for i in range(PositionTwo+1, len(ResultStack)):
            if(ResultStack[i][4]<= 3 and ResultStack[i][4] != -1 and ResultStack[i][5] == 0):
                CollectionSterm.append(ResultStack[i])
            else:
                break
        return CollectionSterm
    def OptiFinalResult(self, RemainMed, ResultStack, Combination, PreStructure, Matrix):
        CollectionSingle = []
        for i in range(len(np.array(RemainMed))):
            CollectionSterm = self.SynCollection(RemainMed[i][7], ResultStack)
            for j in range(len(np.array(CollectionSterm))):
                CollectionSingle.append(CollectionSterm[j])
        if (len(np.array(CollectionSingle)) <= 1):
            Combination = Combination
        else:
            CollecSterm = self.Collection(0, len(np.array(CollectionSingle)) - 1,CollectionSingle)
            Combination = self.DoubleSingleSmallSterm(CollecSterm, PreStructure, Matrix, Combination)
        return Combination
    def ThirdExtendMatrixStermCreate(self, ResultStack, Matrix, PreStructureList):
        MatrixSterm = np.zeros(shape=[len(np.array(ResultStack)), len(np.array(ResultStack))])
        PairedSterm = np.zeros(shape=[len(np.array(ResultStack)), len(np.array(ResultStack))])
        MedOptimal = []
        for i in range(len(MatrixSterm)):
            if(ResultStack[i][3] == 1):
                continue
            else:
                IndexQ = ResultStack[i][1]
                IndexP = ResultStack[i][0]
                for j in range(i):
                    if (ResultStack[j][3] == 1):
                        IndexI = ResultStack[j][0]
                        IndexJ = ResultStack[j][1]
                        AllSterm, flag = self.MaxStem(IndexI, IndexJ, IndexP, IndexQ, Matrix)
                        SelectStemList = self.SelectStem(AllSterm,  flag)
                        if (len(SelectStemList) != 0):
                            OptimalStemList = self.LayerStermCom(SelectStemList, [])
                            numCount = 0
                            StermLength = 0
                            MedExtend = []
                            for num in range(len(np.array(OptimalStemList))):
                                ExtendOptimal = self.IntellExtendStem(Matrix, OptimalStemList[num], PreStructureList, 1, 2)
                                MedExtend.append(ExtendOptimal[0])
                                Rate = self.RateOfSterm(ResultStack[j], ResultStack[i], ExtendOptimal, PreStructureList)
                                if(Rate > numCount):
                                    numCount = Rate
                                    StermLength = ExtendOptimal[0][4]
                            if(StermLength >= 3):
                                MedOptimal.append(MedExtend)
        return MedOptimal
    def MainFuntion(self, FirstLayerStack, SecondaryLayerStack, ThirdLarerStack, PreStructureList, PriList): # The main function for prediction secondary structure.
        Matrix = self.MatrixCreate(PriList, PreStructureList)
        FirstFinal = []
        SecondaryFinal = []
        ThirdFinal = []
        CombinaFinal = []
        if(len(np.array(FirstLayerStack)) != 0): # Using the first class PCRs to get the First class substructure.
            for nnn in range(len(np.array(FirstLayerStack))):
                FirstLayerStack[nnn][7] = nnn
            TwoSuccess = self.MatrixStermCreate(FirstLayerStack, Matrix, PreStructureList) # Get all the exact match PCRs.
            CombinationFirst = self.CorrectSterm(TwoSuccess, Matrix, PreStructureList) # The prediction of first step.
            if (len(np.array(FirstLayerStack)) != 0):
                SecSucc = self.SecondaryMatrixStermCreate(FirstLayerStack, Matrix, PreStructureList)
            else:
                SecSucc = []
            # Get the secondary step prediction results
            if (len(np.array(SecSucc)) != 0):
                CombinationSecondary = self.LayerStermCom(SecSucc,CombinationFirst)
            else:
                CombinationSecondary = CombinationFirst

            ThirdSucc = self.ThirdMatrixStermCreate(FirstLayerStack, Matrix, PreStructureList)
            CombinationThird = self.LayerStermCom(ThirdSucc, CombinationSecondary) # Get the third step prediction results.
            for ii in range(len(np.array(CombinationThird))): # Store the first type of substructure in FirstFinal.
                FirstFinal.append(CombinationThird[ii])
        if(len(np.array(SecondaryLayerStack)) != 0): # Using the secondary class PCRs to get the secondary class substructure.
            for nnn in range(len(np.array(SecondaryLayerStack))):
                SecondaryLayerStack[nnn][7] = nnn
                if(SecondaryLayerStack[nnn][3] == 3):
                    SecondaryLayerStack[nnn][3] = 1
                if(SecondaryLayerStack[nnn][3] == 4):
                    SecondaryLayerStack[nnn][3] = 2
            TwoSuccess2 = self.MatrixStermCreate(SecondaryLayerStack, Matrix, PreStructureList)
            CombinationFirst2 = self.CorrectSterm(TwoSuccess2, Matrix, PreStructureList)
            if (len(np.array(SecondaryLayerStack)) != 0):
                SecSucc2 = self.SecondaryMatrixStermCreate(SecondaryLayerStack, Matrix, PreStructureList)
            else:
                SecSucc2 = []
            if (len(np.array(SecSucc)) != 0):
                CombinationSecondary2 = self.LayerStermCom(SecSucc2,CombinationFirst2)
            else:
                CombinationSecondary2 = CombinationFirst2
            ThirdSucc2 = self.ThirdMatrixStermCreate(SecondaryLayerStack, Matrix, PreStructureList)
            CombinationThird2 = self.LayerStermCom(ThirdSucc2, CombinationSecondary2)
            for ii in range(len(np.array(CombinationThird2))):  # Store the secondary  type of substructure in SecondaryFinal.
                SecondaryFinal.append(CombinationThird2[ii])
        if (len(np.array(ThirdLarerStack)) != 0):
            for nnn in range(len(np.array(ThirdLarerStack))):
                ThirdLarerStack[nnn][7] = nnn
                if(ThirdLarerStack[nnn][3] == 5):
                    ThirdLarerStack[nnn][3] = 1
                if(ThirdLarerStack[nnn][3] == 6):
                    ThirdLarerStack[nnn][3] = 2
            TwoSuccess3 = self.MatrixStermCreate(ThirdLarerStack, Matrix, PreStructureList)
            CombinationFirst3 = self.CorrectSterm(TwoSuccess3, Matrix, PreStructureList)
            if (len(np.array(ThirdLarerStack)) != 0):
                SecSucc3 = self.SecondaryMatrixStermCreate(ThirdLarerStack, Matrix, PreStructureList)
            else:
                SecSucc3 = []
            if (len(np.array(SecSucc)) != 0):
                CombinationSecondary3 = self.LayerStermCom(SecSucc3,CombinationFirst3)
            else:
                CombinationSecondary3 = CombinationFirst3
            ThirdSucc3 = BasisFunction.ThirdMatrixStermCreate(ThirdLarerStack, Matrix, PreStructureList)
            CombinationThird3 = BasisFunction.LayerStermCom(ThirdSucc3, CombinationSecondary3)
            for ii in range(len(np.array(CombinationThird3))):  # Store the secondary  type of substructure in SecondaryFinal.
                ThirdFinal.append(CombinationThird3[ii])
        MedCom = []
        '''
        Combine three substructures to get the final stems combination.
        '''
        if(len(np.array(SecondaryFinal)) != 0):
            for i in range(len(np.array(FirstFinal))):
                for j in range(len(np.array(SecondaryFinal))):
                    TemCombintion = FirstFinal[i]
                    for m in range(len(np.array(SecondaryFinal[j]))):
                        TemCombintion.append(SecondaryFinal[j][m])
                    MedCom.append(TemCombintion)
            Combination = cy.deepcopy(MedCom)
        else:
            Combination = cy.deepcopy(FirstFinal)
        if (len(np.array(ThirdFinal)) != 0):
            for i in range(len(np.array(Combination))):
                for j in range(len(np.array(ThirdFinal))):
                    TemCombintion = Combination[i]
                    for m in range(len(np.array(ThirdFinal[j]))):
                        TemCombintion.append(ThirdFinal[j][m])
                    MedCom.append(TemCombintion)
            Combination = cy.deepcopy(MedCom)
        else:
            Combination = Combination
        for ii in range(len(np.array(Combination))):
            Tem = self.IntellExtendStem(Matrix, np.array(Combination[ii]), PreStructureList, 1, 2)
            CombinaFinal.append(Tem)
        return CombinaFinal