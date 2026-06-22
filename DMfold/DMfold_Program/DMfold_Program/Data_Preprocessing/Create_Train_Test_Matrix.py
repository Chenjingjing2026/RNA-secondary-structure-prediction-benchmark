import numpy as np
import DataMerge as dm
import h5py as h5
import os
import h5py
'''
Converting training data and test data into matrices to facilitate the use of Deep Learning Model.
'''
filePath = '/home/chenjingjing/DATA/data/bprna-new/without_pes/short/bpseq'
# filePath = '/home/chenjingjing/DATA/data/archiveII/shortdata/bpseq'
# filePath = 'OriginalData/TrainData/TrainSet_10' # The path of data(Trainset or TestSet).
def DataCreate():
    nameList = dm.EachFile(filePath) # Get all file names under filepath and return a list.
    SeqList = []
    StrList = []
    StruetureList = []
    PriList = []
    FileIndex = [] # The index of each matrix, which different fragments of the same RNA have the same FileIndex.
    numLength = 300 # The length of each matrix.
    Length = []
    for ctName in range(len(np.array(nameList))): # Processing of each file.
        Pri = []
        List = []
        Seq = []
        Structure = []
        StrListAll = []
        ctPath = dm.FileNmae(filePath, nameList[ctName]) # Get the absolute path of the file.
        fileRead = open(ctPath, mode='r', encoding='UTF-8')
        line = fileRead.readline()
        '''
        Read the file line by line and extract sequence data and structure data to store them in List
        '''
        valid_bases = {'A', 'U', 'G', 'C', 'a', 'u', 'g', 'c'}
        while line and line.split()[0] != '1':
            line = fileRead.readline()
        while line:
            if(len(line.split()) == 3):
            # if (len(line.split()) == 6):
                # list = [line.split()[0], line.split()[1], line.split()[-2], 0]
                # Extract the information
                seq_base = line.split()[1]
                # Check if the base is valid
                if seq_base not in valid_bases:
                    # Skip lines with invalid base letters
                    line = fileRead.readline()
                    continue
                list = [line.split()[0], line.split()[1], line.split()[2], 0]
                List.append(list)
            line = fileRead.readline()
        List = np.array(List)
        # Print out the number of valid data entries in List
        # print(f"File: {nameList[ctName]} - Number of valid entries in List: {len(List)}")
        flag = -1
        Stack = []
        '''
         According to the rules of RNA secondary structure.  
         We convert '(' to 1, ')' to 2, '[' to 3, ']' to 4, '{' to 5  '}' to 6 and '.' to 0.
        '''

        for i in range(len(List)):
            # 处理数据时跳过不符合条件的数据项
            # if List[i][1] not in valid_bases:
            #     continue  # 跳过无效的字母
                # 继续处理有效数据
            if(i == flag):
                Stack.pop()
                if(not Stack):
                    flag = -1
                else:
                    flag = Stack[-1]
            if(List[i][2] != '0' and int(List[i][0]) < int(List[i][2]) and List[i][3] == '0' and int(List[i][2]) < len(List)):
                if(flag < 0):
                    List[i][3] = '1'
                    List[int(List[i][2])-1][3] = '2'
                    flag = int(List[i][2])-1
                    Stack.append(int(List[i][2])-1)
                else:
                    if(int(List[i][2])-1 < flag):
                        List[i][3] = '1'
                        List[int(List[i][2])-1][3] = '2'
                        flag = int(List[i][2])-1
                        if (Stack[-1]  == int(List[i][2]) ):
                            Stack.pop()
                        Stack.append(int(List[i][2]) - 1)
        flag = -1
        Stack = []
        for i in range(len(List)):
            # # 处理数据时跳过不符合条件的数据项
            # if List[i][1] not in valid_bases:
            #     continue  # 跳过无效的字母
            if (i == flag):
                Stack.pop()
                if (not Stack):
                    flag = -1
                else:
                    flag = Stack[-1]
            if (List[i][2] != '0' and int(List[i][0]) < int(List[i][2]) and List[i][3] == '0'  and int(List[i][2]) < len(List)):
                if (flag < 0):
                    List[i][3] = '3'
                    List[int(List[i][2]) - 1][3] = '4'
                    flag = int(List[i][2]) - 1
                    Stack.append(int(List[i][2]) - 1)
                else:
                    if (int(List[i][2]) - 1 < flag):
                        List[i][3] = '3'
                        List[int(List[i][2]) - 1][3] = '4'
                        flag = int(List[i][2]) - 1
                        if (Stack[-1] == int(List[i][2])):
                            Stack.pop()
                        Stack.append(int(List[i][2]) - 1)
        for i in range(len(List)):
            if (List[i][2] != '0' and int(List[i][0]) < int(List[i][2]) and List[i][3] == '0'  and int(List[i][2]) < len(List)):
                List[i][3] = '5'
                List[int(List[i][2]) - 1][3] = '6'
        '''
        When the length of RNA sequences are less than 300. First, we transform RNA sequences and dot_bracket sequences 
        into one-hot according to one-hot coding rule. Secondary, padding zero vectors until the length equal to 300.

        '''
        if(len(List)<= 300):
            Length.append(len(List))
            FileIndex.append(ctName)
            for i in range(len(List)):
                # # 处理数据时跳过不符合条件的数据项
                # if List[i][1] not in valid_bases:
                #     continue  # 跳过无效的字母
                '''
                Convert bases into one-hot encoding.
                '''
                if (List[i][1] == 'A' or List[i][1] == 'a'):
                    Seq.append([1, 0, 0, 0,  0, 0, 1, 0])
                    Pri.append([int(List[i][0]),1,int(List[i][2])])
                elif (List[i][1] == 'U' or List[i][1] == 'u'):
                    Seq.append([0, 0, 1, 0, 1, 0, 0, 0])
                    Pri.append([int(List[i][0]), 2, int(List[i][2])])
                elif (List[i][1] == 'G' or List[i][1] == 'g'):
                    Seq.append([0, 1, 0, 0, 0, 0, 0, 1])
                    Pri.append([int(List[i][0]), 3, int(List[i][2])])
                elif (List[i][1] == 'C' or List[i][1] == 'c'):
                    Seq.append([0, 0, 0, 1, 0, 1, 0, 0])
                    Pri.append([int(List[i][0]), 4, int(List[i][2])])
                # else:
                #     continue  # 跳过非法字母的数据

                if (List[i][2] == '0'):
                    Structure.append([1, 0])
                else:
                    Structure.append([0, 1])
                '''
                Convert dot_bracket sequences into one-hot encoding.
                '''
                if(List[i][3] == '0'):
                    StrListAll.append([0,0, 0,1,0, 0, 0])
                elif(List[i][3] == '1'):
                    StrListAll.append([1,0,0, 0,0, 0,0])
                elif(List[i][3] == '2'):
                    StrListAll.append([0,0,0, 0,0, 0,1])
                elif(List[i][3] == '3'):
                    StrListAll.append([0,1,0, 0,0, 0,0])
                elif(List[i][3] == '4'):
                    StrListAll.append([0,0,0, 0,0, 1,0])
                elif(List[i][3] == '5'):
                    StrListAll.append([0,0,1, 0,0, 0,0])
                else:
                    StrListAll.append([0,0,0, 0,1, 0,0])
            '''
            Padding zero vectors to those matrix which length is less to 300.
            '''
            for j in range(len(List), 300):
                Seq.append([0, 0, 0, 0, 0, 0, 0, 0])
                Structure.append([0, 0])
                StrListAll.append([0,0,0, 0,0, 0,0])
                Pri.append([0, 0 ,0])
            SeqList.append(Seq)
            StrList.append(Structure)
            StruetureList.append(StrListAll)
            PriList.append(Pri)

        else:
            '''
                   When the length of RNA sequences are greater than 300. First intercept those RNA sequences from the beginning into 
                   multi sub-sequences that the length is  300. The overlap length between two consecutive sub-sequences is 200. Secondary,
                   we transform RNA sequences and dot_bracket sequences into one-hot according to one-hot coding rule and padding zero vectors
                   to those sequences which the length is less to 300.

            '''
            count = 0
            '''
            When the length great to 300, intercept the sub sequences.
            '''
            while (len(List) - count * 100 > 300):
                Seq = []
                Structure = []
                StrListAll = []
                Pri = []
                Length.append(numLength)
                FileIndex.append(ctName)
                '''
                The overlap length is 200 between two adjacent sub-sequences.
                '''
                for i in range(count * 100, count * 100 + 300):
                    # # 处理数据时跳过不符合条件的数据项
                    # if List[i][1] not in valid_bases:
                    #     continue  # 跳过无效的字母
                    medPri = []
                    medPri.append(int(List[i][0]))
                    if (List[i][1] == 'A' or List[i][1] == 'a'):
                        Seq.append([1, 0, 0, 0, 0, 0, 1, 0])
                        medPri.append(1)
                    elif (List[i][1] == 'U' or List[i][1] == 'u'):
                        Seq.append([0, 0, 1, 0, 1, 0, 0, 0])
                        medPri.append(2)
                    elif (List[i][1] == 'G' or List[i][1] == 'g'):
                        Seq.append([0, 1, 0, 0, 0, 0, 0, 1])
                        medPri.append(3)
                    # elif (List[i][1] == 'C' or List[i][1] == 'c'):
                    #     Seq.append([0, 0, 0, 1, 0, 1, 0, 0])
                    #     medPri.append(4)
                    # else:
                    #     continue
                    else:
                        Seq.append([0, 0, 0, 1, 0, 1, 0, 0])
                        medPri.append(4)
                    medPri.append(int(List[i][2]))
                    Pri.append(medPri)
                    if (List[i][2] == '0'):
                        Structure.append([1, 0])
                    else:
                        Structure.append([0, 1])
                    if (List[i][3] == '0'):
                        StrListAll.append([0, 0, 0, 1, 0, 0, 0])
                    elif (List[i][3] == '1'):
                        StrListAll.append([1, 0, 0, 0, 0, 0, 0])
                    elif (List[i][3] == '2'):
                        StrListAll.append([0, 0, 0, 0, 0, 0, 1])
                    elif (List[i][3] == '3'):
                        StrListAll.append([0, 1, 0, 0, 0, 0, 0])
                    elif (List[i][3] == '4'):
                        StrListAll.append([0, 0, 0, 0, 0, 1, 0])
                    elif (List[i][3] == '5'):
                        StrListAll.append([0, 0, 1, 0, 0, 0, 0])
                    else:
                        StrListAll.append([0, 0, 0, 0, 1, 0, 0])
                count += 1
                SeqList.append(Seq)
                StrList.append(Structure)
                StruetureList.append(StrListAll)
                PriList.append(Pri)
            Length.append(len(List) - count * 100)
            FileIndex.append(ctName)
            Seq = []
            Structure = []
            StrListAll = []
            Pri = []
            for i in range(count * 100, len(List)):
                # # 处理数据时跳过不符合条件的数据项
                # if List[i][1] not in valid_bases:
                #     continue  # 跳过无效的字母
                medPri = []
                medPri.append(int(List[i][0]))
                if (List[i][1] == 'A' or List[i][1] == 'a'):
                    Seq.append([1, 0, 0, 0, 0, 0, 1, 0])
                    medPri.append(1)
                elif (List[i][1] == 'U' or List[i][1] == 'u'):
                    Seq.append([0, 0, 1, 0, 1, 0, 0, 0])
                    medPri.append(2)
                elif (List[i][1] == 'G' or List[i][1] == 'g'):
                    Seq.append([0, 1, 0, 0, 0, 0, 0, 1])
                    medPri.append(3)
                # elif (List[i][1] == 'C' or List[i][1] == 'c'):
                #     Seq.append([0, 0, 0, 1, 0, 1, 0, 0])
                #     medPri.append(4)
                # else:
                #     continue
                else:
                    Seq.append([0, 0, 0, 1, 0, 1, 0, 0])
                    medPri.append(4)
                medPri.append(int(List[i][2]))
                Pri.append(medPri)
                if (List[i][2] == '0'):
                    Structure.append([1, 0])
                else:
                    Structure.append([0, 1])
                if (List[i][3] == '0'):
                    StrListAll.append([0, 0, 0, 1, 0, 0, 0])
                elif (List[i][3] == '1'):
                    StrListAll.append([1, 0, 0, 0, 0, 0, 0])
                elif (List[i][3] == '2'):
                    StrListAll.append([0, 0, 0, 0, 0, 0, 1])
                elif (List[i][3] == '3'):
                    StrListAll.append([0, 1, 0, 0, 0, 0, 0])
                elif (List[i][3] == '4'):
                    StrListAll.append([0, 0, 0, 0, 0, 1, 0])
                elif (List[i][3] == '5'):
                    StrListAll.append([0, 0, 1, 0, 0, 0, 0])
                else:
                     StrListAll.append([0, 0, 0, 0, 1, 0, 0])
            '''
            For subsequences less than 300 in length, padding zero vectors to them.
            '''
            for j in range(len(List) - count *100, 300):
                Seq.append([0, 0, 0, 0, 0, 0, 0, 0])
                Structure.append([0, 0])
                StrListAll.append([0, 0, 0, 0, 0, 0, 0])
                Pri.append([0, 0, 0])
            SeqList.append(Seq)
            StrList.append(Structure)
            StruetureList.append(StrListAll)
            PriList.append(Pri)
    print("Length of each list:")
    print("SeqList length:", len(SeqList))
    print("StrList length:", len(StrList))
    print("Length length:", len(Length))
    print("StructureList length:", len(StruetureList))
    print("PriList length:", len(PriList))
    print("FileIndex length:", len(FileIndex))
    # print("First element of SeqList:", SeqList[0])
    # 检查每个列表的元素
    # print(f"SeqList[0]: {SeqList[0]}")
    # print(f"Type of SeqList[0]: {type(SeqList[0])}")
    # 检查 SeqList 中的所有子列表的长度是否相同
    # print([len(item) for item in SeqList])

    # print(f"StrList[0]: {StrList[0]}")
    # print(f"Type of SeqList[0]: {type(StrList[0])}")
    # return np.array(SeqList, dtype=object), np.array(StrList, dtype=object), np.array(Length, dtype=object), np.array(
        # StruetureList, dtype=object), np.array(PriList, dtype=object), np.array(FileIndex, dtype=object)

    return np.array(SeqList), np.array(StrList), np.array(Length), np.array(StruetureList), np.array(PriList), np.array(FileIndex)

def AllDataWrite():

    SeqList, StrList, Length, StructureList, PriList, FileIndex = DataCreate()

    '''
    Store the transformed matrix data as H5 file
    '''

    # 检查路径是否存在，不存在则创建
    dir_path = '/home/chenjingjing/Models/DMfold/myData/bprna_new'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 然后再创建文件
    AllDataFile = h5.File(os.path.join(dir_path, 'test.h5'), 'w')
    # AllDataFile = h5.File('myData/RNAStrAlign/test.h5', 'w')
    AllDataFile.create_dataset('SeqList', data=SeqList)
    AllDataFile.create_dataset('FairList', data=StrList)
    AllDataFile.create_dataset('Length', data=Length)
    AllDataFile.create_dataset('StructureList', data=StructureList)
    AllDataFile.create_dataset('PriList', data=PriList)
    AllDataFile.create_dataset('FileIndex', data=FileIndex)
    AllDataFile.close()

def DATAwrite():

    # 假设 DataCreate() 返回的数据是列表形式，且每条数据的索引对应
    SeqList, StrList, Length, StructureList, PriList, FileIndex = DataCreate()

    # 检查路径是否存在，不存在则创建
    dir_path = '/home/chenjingjing/Models/DMfold/myData/bprna_new'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 遍历每一条数据
    for i in range(len(SeqList)):
        # 为每条数据创建一个 H5 文件
        file_name = 'data_{}.h5'.format(i)  # 使用 str.format() 方法
        file_path = os.path.join(dir_path, file_name)

        # 创建 H5 文件并存储数据
        with h5py.File(file_path, 'w') as data_file:
            data_file.create_dataset('SeqList', data=SeqList[i])
            data_file.create_dataset('FairList', data=StrList[i])
            data_file.create_dataset('Length', data=Length[i])
            data_file.create_dataset('StructureList', data=StructureList[i])
            data_file.create_dataset('PriList', data=PriList[i])
            data_file.create_dataset('FileIndex', data=FileIndex[i])

    print("All data has been saved to {}".format(dir_path))

#多条数据
AllDataWrite()
#单条数据
# DATAwrite()






