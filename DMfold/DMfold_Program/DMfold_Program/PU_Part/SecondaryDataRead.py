import h5py as h5
import numpy as np
import h5py
def MatureReadTrainDataBatch():
    Train = h5.File('/home/chenjingjing/Models/DMfold/myData/bprna_1m/train.h5', 'r')
    with h5py.File('/home/chenjingjing/Models/DMfold/myData/bprna_1m/train.h5', 'r') as f:
        # 打印文件中的所有对象（数据集）
        print('Train Keys')
        print(list(f.keys()))
    # Train = h5.File('Deep_Model_Matrix_Data/TrainData_1.h5', 'r')   # Reading the train data.
    train_x = Train['SeqList'][:]   # Get the RNA sequences data.
    train_y = Train['StructureList'][:] # Get the RNA structure data.
    train_l = Train['Length'][:]    # Get the real length of each matrix.
    Train.close()
    Test = h5.File('/home/chenjingjing/Models/DMfold/myData/bprna_new/test.h5', 'r')
    # Test = h5.File('Deep_Model_Matrix_Data/TestData.h5', 'r')   # Reading the test data.
    with h5py.File('/home/chenjingjing/Models/DMfold/myData/bprna_new/test.h5', 'r') as f:
        # 打印文件中的所有对象（数据集）
        print('Test Keys')
        print(list(f.keys()))
    test_x = Test['SeqList'][:] # Get the RNA sequences data.
    test_y = Test['StructureList'][:]   # Get the RNA structure data.
    test_l = Test['Length'][:]  # Get the real length of each matrix.
    PriTestList = Test['PriList'][:]
    PriFileTestIndex = Test['FileIndex'][:]
    Test.close()
    return np.array(train_x),np.array(train_y), np.array(train_l), np.array(test_x), np.array(test_y), np.array(test_l), np.array(PriTestList),np.array(PriFileTestIndex)
'''
In order to set all the prediction results of padding vector to zero vector. According to the real length of 
matrix to creat a matrix, which set the local of real vector is [1,1,1,1,1,1,1] and padding vector is [0,0,0,0,0,0].
Hence, multiplying those two matrix  to setting the prediction results of the padding vectors to zero vectors.
'''
def Accuracy(train_l, test_l):  # train_l is the real length of train data, test_l is the real length of test data.
    AccuracyTrain = []
    AccurayTest = []
    for i in range(len(train_l)):
        Accury_train = []
        for j in range(300):
            if(j<train_l[i]):
                Accury_train.append([1, 1, 1, 1, 1, 1, 1])
            else:
                Accury_train.append([0, 0, 0, 0, 0, 0, 0])
        AccuracyTrain.append(Accury_train)
    for i in range(len(test_l)):
        Accur_test = []
        for j in range(300):
            if(j<test_l[i]):
                Accur_test.append([1, 1, 1, 1, 1, 1, 1])
            else:
                Accur_test.append([0, 0, 0, 0, 0, 0, 0])
        AccurayTest.append(Accur_test)
    return np.array(AccuracyTrain), np.array(AccurayTest)
'''
In order to computation the real prediction result, it is necessary to return the real average length of the batch.
'''
def Alllength(train_length):
    Tcount = 0
    for i in range(len(train_length)):
        Tcount += train_length[i]
    return Tcount/len(train_length)
