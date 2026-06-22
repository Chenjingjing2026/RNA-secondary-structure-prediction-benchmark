import tensorflow as tf  # Improt the module of tensorflow
import numpy as np # Improt the module of numpy
import SecondaryDataRead as sd # Improt the program of SecondaryDataRead
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import LSTM, Bidirectional, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import InputSpec

import h5py as h5

HIDDEN_SIZE = 300 # The number of LSTM hidden node.
NUM_LAYERS = 3  # The number of LSTM layer.
NUM_INPUT = 8   # The number of node in each step.
N_CLASSES = 7   # The number of the prediction result in each base.
TRAIN_BATCH_SIZE = 16   # train batch size
TEST_BATCH_SIZE = 316  # testing data size
TESTTRAIN_BATCH_SIZE = 2521 # training data size
TRAIN_NUM_STEP = 300    # the number of step in LSTM.
LSTM_KEEP_PROB = 0.9  # the rate of dropout in LSTM(encoder).
FULL_LAYER_DROPOUT = 0.9 # the rate of dropout in fully connected layer(decoder).

class PaperModel(tf.Module):  # Inherit from tf.Module
# class PaperModel(object): # Create the class of model.

    def __init__(self, is_training ,batch_size, num_steps, n_input, n_classes):# # The initlization function. is_training is to determination which the set is training_set，batch_size：the size of batch，num_steps：the number of steps，n_input：the number of input nodes of each step，n_classes：number of categories.
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.n_input = n_input
        print('self.n_input',self.n_input)
        self.n_classes = n_classes
        # Define the model's layers
        self.dense = tf.keras.layers.Dense(n_classes)

        # Use tf.keras.Input to define the shape of input data
        self.input_data = tf.keras.Input(shape=(num_steps, n_input))
        self.targets = tf.keras.Input(shape=(num_steps, n_classes))

        # self.input_data = tf.placeholder(tf.float32, [None, num_steps, n_input])  # Definition the shape of input data: [None, num_steps, n_input].
        # self.targets = tf.placeholder(tf.float32, [None, num_steps, n_classes])  # Definition the shape of output data: [None, num_steps, n_classes]
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0 # # If in the train: dropout_keep_prob=0.9，If in the test:dropout_keep_prob=1.0
        # # Definition the cell of forward LSTM network.
        # lstm_fw_cell = [
        #     tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=dropout_keep_prob)
        #     for _ in range(NUM_LAYERS)]
        # # Definition the cell of backward LSTM network.
        # lstm_bw_cell = [
        #     tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=dropout_keep_prob)
        #     for _ in range(NUM_LAYERS)]
        # fw_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_fw_cell)
        # bw_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_bw_cell)
        #
        # # Definition the cell of forward LSTM network.
        # lstm_fw_cell = [
        #     tf.keras.layers.LSTMCell(HIDDEN_SIZE)  # Replaced BasicLSTMCell with LSTMCell
        #     for _ in range(NUM_LAYERS)
        # ]
        #
        # # Definition the cell of backward LSTM network.
        # lstm_bw_cell = [
        #     tf.keras.layers.LSTMCell(HIDDEN_SIZE)  # Replaced BasicLSTMCell with LSTMCell
        #     for _ in range(NUM_LAYERS)
        # ]
        #
        # # Apply dropout for each LSTM cell if training
        # if is_training:
        #     lstm_fw_cell = [
        #         tf.keras.layers.Dropout(dropout_keep_prob)(cell)
        #         for cell in lstm_fw_cell
        #     ]
        #     lstm_bw_cell = [
        #         tf.keras.layers.Dropout(dropout_keep_prob)(cell)
        #         for cell in lstm_bw_cell
        #     ]
        #
        # # Combine forward and backward LSTM cells
        # fw_cell = tf.keras.layers.RNN(lstm_fw_cell, return_sequences=True, return_state=True)
        # bw_cell = tf.keras.layers.RNN(lstm_bw_cell, return_sequences=True, return_state=True)

        # 定义前向和后向 LSTM 单元
        self.lstm_fw_cells = [
            tf.keras.layers.LSTMCell(units=HIDDEN_SIZE, dropout=1 - dropout_keep_prob)
            for _ in range(NUM_LAYERS)
        ]
        self.lstm_bw_cells = [
            tf.keras.layers.LSTMCell(units=HIDDEN_SIZE, dropout=1 - dropout_keep_prob)
            for _ in range(NUM_LAYERS)
        ]

        # 使用 MultiRNNCell 堆叠多层 LSTM
        self.fw_cell = tf.keras.layers.StackedRNNCells(self.lstm_fw_cells)
        self.bw_cell = tf.keras.layers.StackedRNNCells(self.lstm_bw_cells)

        self.input_dataone = tf.transpose(self.input_data, [1, 0, 2])
        self.input_dataone = tf.reshape(self.input_dataone, [-1, NUM_INPUT])
        self.input_dataone = tf.split(self.input_dataone,TRAIN_NUM_STEP)  # Transform the input data into the LSTM data.


        # 使用 Bidirectional 包装器
        self.bidirectional_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(self.fw_cell, return_sequences=True, return_state=True),
            backward_layer=tf.keras.layers.RNN(self.bw_cell, return_sequences=True, return_state=True, go_backwards=True)
            # 设置 go_backwards=True
        )

        # 输入数据不需要手动转置和拆分，Keras 层会自动处理
        # outputs, forward_state, backward_state = self.bidirectional_rnn(self.input_data)
        # Assuming you're using LSTM cells, the initial state should be a tuple of (h, c)
        # self.forward_initial_state = [tf.zeros([HIDDEN_SIZE, TRAIN_NUM_STEP]) for _ in range(3)]
        # self.backward_initial_state = [tf.zeros([HIDDEN_SIZE, TRAIN_NUM_STEP]) for _ in range(3)]

        # # Combine them into a list
        # self.initial_state = self.forward_initial_state + self.backward_initial_state
        # zeros_2d = tf.zeros([HIDDEN_SIZE, TRAIN_NUM_STEP])
        # zeros_3d = tf.expand_dims(zeros_2d, axis=0)
        # zeros_3d = tf.transpose(zeros_3d, perm=[0, 2, 1])
        # self.initial_state = tf.tile(zeros_3d, multiples=[batch_size, 1, 1])
        self.forward_initial_state = [tf.zeros([HIDDEN_SIZE, TRAIN_NUM_STEP]) for _ in range(3)]
        self.backward_initial_state = [tf.zeros([HIDDEN_SIZE, TRAIN_NUM_STEP])]

        # self.initial_state = tf.keras.utils.ListWrapper(tf.zeros([HIDDEN_SIZE, TRAIN_NUM_STEP]))
        # 创建三个状态，每个状态大小是 (batch_size, state_size)

        # self.initial_state = [[spec.shape[0], spec.shape[1]] for spec in self.initial_state]
        # 初始化状态，使用 Tensor 来初始化
        # initial_state = [tf.zeros([HIDDEN_SIZE, TRAIN_NUM_STEP])] * 3  # 创建三个状态，每个状态大小是 (batch_size, state_size)
        #
        # # 使用 ListWrapper 将这些状态包装起来
        # self.initial_state = tf.nest.pack_sequence_as(self.initial_state, self.initial_state)
        print("Initial state type:", type(self.initial_state))
        # print("Initial state:", self.initial_state)
        outputs, forward_state, backward_state, *additional_states = self.bidirectional_rnn(self.input_data, initial_state=zeros_3d)

        # 调用 Bidirectional RNN

        # outputs, *states = self.bidirectional_rnn(self.input_data)

        # outputs, *states = self.bidirectional_rnn(self.input_dataone, initial_state=None)

        # # 处理最终状态
        # # 对于 LSTM，states 包含前向和后向的最终状态（h 和 c）
        # forward_state_h, forward_state_c = states[:2]  # 前向的 h 和 c
        # backward_state_h, backward_state_c = states[2:]  # 后向的 h 和 c
        #
        # # 如果需要将前向和后向的最终状态合并
        # final_state_h = tf.concat([forward_state_h, backward_state_h], axis=-1)
        # final_state_c = tf.concat([forward_state_c, backward_state_c], axis=-1)


        # outputs, state_fw, state_bw = tf.keras.layers.RNN(
        #     tf.keras.layers.StackedRNNCells([fw_cell, bw_cell]),
        #     return_sequences=True,
        #     return_state=True
        # )(self.input_dataone)

        # outputs, _, _ = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, self.input_dataone,dtype=tf.float32)  # outputs is Lstm output, the shape of outputs is [num_steps, batch_size, 2*HIDDEN]

        # Define a Bidirectional LSTM layer
        # bi_lstm_layer = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))
        #
        # # Apply the Bidirectional LSTM to the input data
        # outputs = bi_lstm_layer(self.input_dataone)  # outputs is LSTM output, shape: [batch_size, num_steps, 2*HIDDEN]

        # 权重和偏置
        self.weightone = tf.Variable(tf.random.normal([2 * HIDDEN_SIZE, HIDDEN_SIZE]), name="weightone")
        self.biasone = tf.Variable(tf.zeros([HIDDEN_SIZE]), name="biasone")

        self.weighttwo = tf.Variable(tf.random.normal([HIDDEN_SIZE, HIDDEN_SIZE // 2]), name="weighttwo")
        self.biastwo = tf.Variable(tf.zeros([HIDDEN_SIZE // 2]), name="biastwo")

        self.weightthree = tf.Variable(tf.random.normal([HIDDEN_SIZE // 2, N_CLASSES]), name="weightthree")
        self.biasthree = tf.Variable(tf.zeros([N_CLASSES]), name="biasthree")

        # weightone = tf.get_variable("weightone", [2*HIDDEN_SIZE, HIDDEN_SIZE])
        # biasone = tf.get_variable('biasone', [HIDDEN_SIZE])
        # weighttwo = tf.get_variable("weighttwo", [HIDDEN_SIZE, HIDDEN_SIZE/2])
        # biastwo = tf.get_variable('biastwo', [HIDDEN_SIZE/2])
        # weightthree = tf.get_variable("weight", [HIDDEN_SIZE/2, N_CLASSES])
        # biasthree = tf.get_variable('bias', [N_CLASSES])
        self.full_dropout_keep_prob = FULL_LAYER_DROPOUT if is_training else 1.0
        self.layer1 = tf.nn.relu(tf.matmul(outputs, self.weightone) + self.biasone)
        self.layer1 = tf.nn.dropout(self.layer1, self.full_dropout_keep_prob)
        self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.weighttwo) + self.biastwo)
        self.layer2 = tf.nn.dropout(self.layer2, self.full_dropout_keep_prob)
        self.logits = tf.nn.softmax(tf.matmul(self.layer2, self.weightthree) + self.biasthree) # Calculate prediction results.


    def __call__(self, input_data, training=False):
        """
        Define the forward pass of the model
        :param input_data: input tensor, shape [batch_size, num_steps, n_input]
        :param training: boolean flag indicating whether the model is in training mode (to enable dropout)
        :return: logits tensor, shape [batch_size, num_steps, n_classes]
        """
        return self.call(input_data, training)


    def call(self, input_data, training=False):
        """
        Define the forward pass of the model
        :param input_data: input tensor, shape [batch_size, num_steps, n_input]
        :param training: boolean flag indicating whether the model is in training mode (to enable dropout)
        :return: logits tensor, shape [batch_size, num_steps, n_classes]
        """
        dropout_keep_prob = LSTM_KEEP_PROB if training else 1.0  # Set dropout probability for training

        # Prepare the input data for the LSTM layer
        self.input_dataone = tf.transpose(input_data, [1, 0, 2])  # Change shape to [num_steps, batch_size, n_input]
        self.input_dataone = tf.reshape(self.input_dataone, [-1, self.n_input])  # Reshape to [-1, n_input]
        self.input_dataone = tf.split(self.input_dataone, self.num_steps)  # Split the input into time steps

        # Bidirectional RNN using stacked LSTM cells
        self.bidirectional_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(self.fw_cell, return_sequences=True, return_state=True),
            backward_layer=tf.keras.layers.RNN(self.bw_cell, return_sequences=True, return_state=True, go_backwards=True)
        )

        # # Assuming you're using LSTM cells, the initial state should be a tuple of (h, c)
        # forward_initial_state = [tf.zeros([self.batch_size, self.num_steps]), tf.zeros([self.batch_size, self.num_steps])]
        # backward_initial_state = [tf.zeros([self.batch_size, self.num_steps]), tf.zeros([self.batch_size, self.num_steps])]
        #
        # # Combine them into a list
        # initial_state = forward_initial_state + backward_initial_state

        # Now pass it to the Bidirectional layer
        outputs, forward_state, backward_state, *additional_states = self.bidirectional_rnn(self.input_dataone,
                                                                                            initial_state=self.initial_state)

        # 运行 Bidirectional RNN
        # outputs, *states = self.bidirectional_rnn(self.input_dataone, initial_state=None)  # Ensure no initial_state is passed
        # return outputs  # Proceed with other layers as needed
        outputs, forward_state, backward_state, *additional_states = self.bidirectional_rnn(self.input_dataone)
        # outputs, *states = self.bidirectional_rnn(self.input_dataone)

        # Forward pass through the fully connected layers
        self.full_dropout_keep_prob = FULL_LAYER_DROPOUT if training else 1.0
        self.layer1 = tf.nn.relu(tf.matmul(outputs, self.weightone) + self.biasone)
        self.layer1 = tf.nn.dropout(self.layer1, self.full_dropout_keep_prob)
        self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.weighttwo) + self.biastwo)
        self.layer2 = tf.nn.dropout(self.layer2, self.full_dropout_keep_prob)
        self.logits = tf.nn.softmax(tf.matmul(self.layer2, self.weightthree) + self.biasthree)  # Final output

        return self.logits


# def Test(session, test_x, test_y,model):
#     logits = session.run([model.logits],feed_dict={model.input_data: test_x, model.targets: test_y})
#     return logits
# def TestTrain(session, test_x, test_y,model):
#     logits = session.run([model.logits],feed_dict={model.input_data: test_x, model.targets: test_y})
#     return logits

def Test(test_x, test_y, model):
    # 直接调用模型的前向传播方法
    logits = model(test_x, training=True)  # training=False 表示不启用 Dropout 等训练特有的行为
    return logits

def TestTrain(test_x, test_y, model):
    # 直接调用模型的前向传播方法
    logits = model(test_x, training=True)  # training=False 表示不启用 Dropout 等训练特有的行为
    return logits

if __name__ == "__main__":

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # Instantiate PaperModel with initializer
    test_model = PaperModel(True, TEST_BATCH_SIZE, TRAIN_NUM_STEP, NUM_INPUT, N_CLASSES)
    testTrain_model = PaperModel(True, TESTTRAIN_BATCH_SIZE, TRAIN_NUM_STEP, NUM_INPUT, N_CLASSES)
    # with tf.variable_scope("Paper_Model", reuse=None, initializer=initializer):
    #     test_model = PaperModel(False, TEST_BATCH_SIZE, TRAIN_NUM_STEP, NUM_INPUT, N_CLASSES)
    # with tf.variable_scope("Paper_Model", reuse=True, initializer=initializer):
    #     testTrain_model = PaperModel(False, TESTTRAIN_BATCH_SIZE, TRAIN_NUM_STEP, NUM_INPUT, N_CLASSES)
    train_x, train_y, train_l, test_x, test_y, test_l = sd.MatureReadTrainDataBatch() # Get the train and test data.
    AccuracyTrain, AccurayTest = sd.Accuracy(train_l, test_l)
    AccuracyTrain = np.reshape(AccuracyTrain, [-1, 7])
    AccurayTest = np.reshape(AccurayTest, [-1, 7])
    # saver = tf.train.Saver(max_to_keep=200)
    # 使用 tf.train.Checkpoint 替代 tf.train.Saver
    checkpoint = tf.train.Checkpoint(model=test_model)  # test_model 是你的模型实例
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory='Model', max_to_keep=200)
    checkpoint_manager.save()
    print("Checkpoint saved at:",{checkpoint_manager.latest_checkpoint})
    checkpoint.restore('Model/ckpt-1').expect_partial()

    # checkpoint.restore(
    #     '/home/chenjingjing/Models/DMfold/DMfold_Program/DMfold_Program/PU_Part/Model/model-50.ckpt').expect_partial()
    # checkpoint.restore('/home/chenjingjing/Models/DMfold/DMfold_Program/DMfold_Program/PU_Part/Model/model-50.ckpt').expect_partial()

    # 获取测试数据的预测结果
    Testlogits = Test(test_x, test_y, test_model)  # 不需要 sess
    Trainlogits = TestTrain(train_x, train_y, testTrain_model)  # 不需要 sess
    Testlogits = np.reshape(Testlogits, [-1, 7])
    Trainlogits = np.reshape(Trainlogits, [-1, 7])
    Testlogits = Testlogits*AccurayTest # Remove the prediction of padding bases.
    Trainlogits = Trainlogits*AccuracyTrain # Remove the prediction of padding bases.
    test_y = np.reshape(test_y, [-1, 7])
    Test =tf.argmax(Testlogits, 1)
    Train = tf.argmax(Trainlogits, 1)
    test_y = tf.argmax(test_y, 1)
    TestlogitsNew  = []
    TrainlogitsNew = []
    # Encode prediction results as one-hot vectors
    for i in range(len(Test)):
        if (Test[i] == 0):
            TestlogitsNew.append([1, 0, 0, 0, 0, 0, 0])
        elif (Test[i] == 1):
            TestlogitsNew.append([0, 1, 0, 0, 0, 0, 0])
        elif (Test[i] == 2):
            TestlogitsNew.append([0, 0, 1, 0, 0, 0, 0])
        elif (Test[i] == 3):
            TestlogitsNew.append([0, 0, 0, 1, 0, 0, 0])
        elif (Test[i] == 4):
            TestlogitsNew.append([0, 0, 0, 0, 1, 0, 0])
        elif (Test[i] == 5):
            TestlogitsNew.append([0, 0, 0, 0, 0, 1, 0])
        else:
            TestlogitsNew.append([0, 0, 0, 0, 0, 0, 1])
    for i in range(len(Train)):
        if (Train[i] == 0):
            TrainlogitsNew.append([1, 0, 0, 0, 0, 0, 0])
        elif (Train[i] == 1):
            TrainlogitsNew.append([0, 1, 0, 0, 0, 0, 0])
        elif (Train[i] == 2):
            TrainlogitsNew.append([0, 0, 1, 0, 0, 0, 0])
        elif (Train[i] == 3):
            TrainlogitsNew.append([0, 0, 0, 1, 0, 0, 0])
        elif (Train[i] == 4):
            TrainlogitsNew.append([0, 0, 0, 0, 1, 0, 0])
        elif (Train[i] == 5):
            TrainlogitsNew.append([0, 0, 0, 0, 0, 1, 0])
        else:
            TrainlogitsNew.append([0, 0, 0, 0, 0, 0, 1])
    TestlogitsNew = np.array(TestlogitsNew)
    TrainlogitsNew = np.array(TrainlogitsNew)
    TestlogitsNew = TestlogitsNew*AccurayTest
    TrainlogitsNew = TrainlogitsNew*AccuracyTrain
    TestlogitsNew = np.reshape(TestlogitsNew, [-1, 300, 7])
    TrainlogitsNew = np.reshape(TrainlogitsNew, [-1, 300, 7])
    # Store prediction results in h5 file.
    TrainPreStr = h5.File('Saver_Result/First/TrainPre.h5', 'w')
    TestPreStr = h5.File('Saver_Result/First/Test_RNAPre.h5', 'w')
    TrainPreStr.create_dataset('PreStructure', data=TrainlogitsNew)
    TestPreStr.create_dataset('PreStructure', data=TestlogitsNew)
    TrainPreStr.close()
    TestPreStr.close()

    # with tf.Session() as sess:
    #     # 恢复模型
    #     checkpoint.restore('Model/First/model-50.ckpt').expect_partial()
    #     # saver.restore(sess, 'Model/First/model-50.ckpt') # Restore the record model
    #     Testlogits = Test(sess, test_x, test_y, test_model) # Get the prediction results of test data.
    #     Trainlogits = TestTrain(sess, train_x, train_y,  testTrain_model) # Get the prediction results of train data.
    #     Testlogits = np.reshape(Testlogits, [-1, 7])
    #     Trainlogits = np.reshape(Trainlogits, [-1, 7])
    #     Testlogits = Testlogits*AccurayTest # Remove the prediction of padding bases.
    #     Trainlogits = Trainlogits*AccuracyTrain # Remove the prediction of padding bases.
    #     test_y = np.reshape(test_y, [-1, 7])
    #     Test =tf.argmax(Testlogits, 1)
    #     Train = tf.argmax(Trainlogits, 1)
    #     test_y = tf.argmax(test_y, 1)
    #     TestlogitsNew  = []
    #     TrainlogitsNew = []
    #     # Encode prediction results as one-hot vectors
    #     for i in range(len(Test)):
    #         if (Test[i] == 0):
    #             TestlogitsNew.append([1, 0, 0, 0, 0, 0, 0])
    #         elif (Test[i] == 1):
    #             TestlogitsNew.append([0, 1, 0, 0, 0, 0, 0])
    #         elif (Test[i] == 2):
    #             TestlogitsNew.append([0, 0, 1, 0, 0, 0, 0])
    #         elif (Test[i] == 3):
    #             TestlogitsNew.append([0, 0, 0, 1, 0, 0, 0])
    #         elif (Test[i] == 4):
    #             TestlogitsNew.append([0, 0, 0, 0, 1, 0, 0])
    #         elif (Test[i] == 5):
    #             TestlogitsNew.append([0, 0, 0, 0, 0, 1, 0])
    #         else:
    #             TestlogitsNew.append([0, 0, 0, 0, 0, 0, 1])
    #     for i in range(len(Train)):
    #         if (Train[i] == 0):
    #             TrainlogitsNew.append([1, 0, 0, 0, 0, 0, 0])
    #         elif (Train[i] == 1):
    #             TrainlogitsNew.append([0, 1, 0, 0, 0, 0, 0])
    #         elif (Train[i] == 2):
    #             TrainlogitsNew.append([0, 0, 1, 0, 0, 0, 0])
    #         elif (Train[i] == 3):
    #             TrainlogitsNew.append([0, 0, 0, 1, 0, 0, 0])
    #         elif (Train[i] == 4):
    #             TrainlogitsNew.append([0, 0, 0, 0, 1, 0, 0])
    #         elif (Train[i] == 5):
    #             TrainlogitsNew.append([0, 0, 0, 0, 0, 1, 0])
    #         else:
    #             TrainlogitsNew.append([0, 0, 0, 0, 0, 0, 1])
    #     TestlogitsNew = np.array(TestlogitsNew)
    #     TrainlogitsNew = np.array(TrainlogitsNew)
    #     TestlogitsNew = TestlogitsNew*AccurayTest
    #     TrainlogitsNew = TrainlogitsNew*AccuracyTrain
    #     TestlogitsNew = np.reshape(TestlogitsNew, [-1, 300, 7])
    #     TrainlogitsNew = np.reshape(TrainlogitsNew, [-1, 300, 7])
    #     # Store prediction results in h5 file.
    #     TrainPreStr = h5.File('Saver_Result/First/TrainPre.h5', 'w')
    #     TestPreStr = h5.File('Saver_Result/First/Test_RNAPre.h5', 'w')
    #     TrainPreStr.create_dataset('PreStructure', data=TrainlogitsNew)
    #     TestPreStr.create_dataset('PreStructure', data=TestlogitsNew)
    #     TrainPreStr.close()
    #     TestPreStr.close()
