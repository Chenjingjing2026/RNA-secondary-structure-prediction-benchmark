import h5py as h5
import os
import h5py
import numpy as np

class DataRead:
    def PriArchiveData(self): # Read the real data to calculate the accuracy of the prediction results
        PriTestFile = h5.File('/home/chenjingjing/Models/DMfold/myData/Rivas/TestSetA.h5', 'r')
        PriTestSeqList = PriTestFile['SeqList'][:]
        PriTestStructureList = PriTestFile['StructureList'][:]
        PriTestLengthList = PriTestFile['Length'][:]
        PriTestList = PriTestFile['PriList'][:]
        PriFileTestIndex = PriTestFile['FileIndex'][:]
        # Return the reading data.
        return PriTestSeqList, PriTestStructureList, PriTestLengthList, PriTestList, PriFileTestIndex  # 数据类型均为<class 'numpy.ndarray'>
    def PreStructureData(self): # Read the prediction results
        # PreSecStructureTest = h5.File('Saver_Result/First/Test_RnasepPre.h5', 'r')
        PreSecStructureTest = h5.File('/home/chenjingjing/Models/DMfold/Saver_Result/Rivas/Test_RNAPre.h5', 'r')
        PreStructureTest = PreSecStructureTest['PreStructure'][:]
        PreSecStructureTest.close()
        return PreStructureTest


    # def PriArchiveData(self):
    #     # 文件夹路径
    #     folder_path = '/home/chenjingjing/Models/DMfold/myData/Rivas/test'
    #
    #     # 初始化空列表，用于存储所有文件的数据
    #     all_seq_list = []
    #     all_structure_list = []
    #     all_length_list = []
    #     all_pri_list = []
    #     all_file_index = []
    #
    #     # 遍历文件夹中的所有 H5 文件
    #     for file_name in os.listdir(folder_path):
    #         if file_name.endswith('.h5'):  # 确保只处理 H5 文件
    #             file_path = os.path.join(folder_path, file_name)
    #
    #             # 打开 H5 文件并读取数据
    #             with h5py.File(file_path, 'r') as PriTestFile:
    #                 PriTestSeqList = PriTestFile['SeqList'][:]
    #                 PriTestStructureList = PriTestFile['StructureList'][:]
    #                 PriTestLengthList = PriTestFile['Length'][:]
    #                 PriTestList = PriTestFile['PriList'][:]
    #                 PriFileTestIndex = PriTestFile['FileIndex'][:]
    #
    #                 # 将数据添加到总列表中
    #                 all_seq_list.append(PriTestSeqList)
    #                 all_structure_list.append(PriTestStructureList)
    #                 all_length_list.append(PriTestLengthList)
    #                 all_pri_list.append(PriTestList)
    #                 all_file_index.append(PriFileTestIndex)
    #
    #     # 将列表转换为 NumPy 数组（如果需要）
    #     all_seq_list = np.concatenate(all_seq_list, axis=0)
    #     all_structure_list = np.concatenate(all_structure_list, axis=0)
    #     all_length_list = np.concatenate(all_length_list, axis=0)
    #     all_pri_list = np.concatenate(all_pri_list, axis=0)
    #     all_file_index = np.concatenate(all_file_index, axis=0)
    #
    #     # 返回读取的数据
    #     return all_seq_list, all_structure_list, all_length_list, all_pri_list, all_file_index