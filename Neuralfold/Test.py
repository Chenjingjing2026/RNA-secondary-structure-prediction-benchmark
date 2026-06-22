import numpy as np
import Config
import Recursive
import chainer.links as L
import chainer.functions as F
import Inference
import SStruct
import Deepnet
import Evaluate
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
from time import time
# from multiprocessing import Pool
# import multiprocessing as multi
maximum_slots = Config.maximum_slots
batch_num = Config.batch_num

class Test:
    def __init__(self, args):
        # print('test_file',args.test_file)
        sstruct = SStruct.SStruct(args.test_file, False)
        if args.bpseq:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_BPseq()
            # print('self.structure_set',self.structure_set)
        else:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_FASTA()

        # self.model = Deepnet.Deepnet(200, 50, False, "sigmoid")
        if args.learning_model == "recursive":
            self.model = Recursive.Recursive_net(args.hidden_insideoutside, args.hidden2_insideoutside, args.feature,
                                                 args.hidden_marge, args.hidden2_marge, args.activation_function)
        elif args.learning_model == "deepnet":
            self.model = Deepnet.Deepnet(args.hidden1, args.hidden2, args.hidden3, args.activation_function)
        else:
            print("unexpected network")

        # if args.Parameters:
        #     serializers.load_npz(args.Parameters.name, self.model)
        # else:
        serializers.load_npz("/home/chenjingjing/Models/Neuralfold/result/bprna_1m/bprna_1m.data", self.model)

        self.neighbor = 40
        self.feature = 80
        self.activation_function = "sigmoid"
        self.ipknot = args.ipknot
        self.test_file = args.test_file
        self.gamma = 2**(args.gamma)+1
        self.args =args

    def test(self):
        predicted_structure_set = []
        start_test = time()
        output_data = []
        for name,seq in zip(self.name_set, self.seq_set):
            # print(name)
            inference = Inference.Inference(seq,self.feature, self.activation_function,False)
            if self.args.learning_model == "recursive":
                predicted_BP = inference.ComputeInsideOutside(self.model)
            elif self.args.learning_model == "deepnet":
                predicted_BP, predicted_UP_left, predicted_UP_right = inference.ComputeNeighbor(self.model, self.neighbor)
            else:
                print("unexpected network")
            # predicted_BP, predicted_UP_left, predicted_UP_right = inference.ComputeNeighbor(self.model, self.neighbor)
            predicted_structure=inference.ComputePosterior(predicted_BP, 0, self.ipknot, self.gamma, "Test",np.zeros((len(seq),len(seq)), dtype=np.float32))
            predicted_structure_set.append(predicted_structure)

            # 将每个 name, seq 和 predicted_structure 格式化并添加到 output_data 列表中
            output_data.append(f"> {name}\n{seq}\n{predicted_structure}\n")

        output_file = '/home/chenjingjing/Models/Neuralfold/result/bprna_1m/bprna_new.txt'  # 你可以根据需要修改文件名
        # 在所有处理完毕后，一次性写入文件
        with open(output_file, 'w') as f:
            f.writelines(output_data)

        # print('Test time: ' + str(time() - start_test) + ' sec')s
        print(' test time: '+str(time() - start_test)+'sec')

        if self.structure_set:
            evaluate = Evaluate.Evaluate(predicted_structure_set, self.structure_set)
            # print('evaluate',evaluate)
            Sensitivity, PPV, F_value, MCC = evaluate.getscore()



            return Sensitivity, PPV, F_value, MCC
        else:
            return 0,0,0
