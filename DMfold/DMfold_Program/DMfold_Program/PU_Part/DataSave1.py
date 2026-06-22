import tensorflow as tf  # Improt the module of tensorflow
import numpy as np # Improt the module of numpy
import SecondaryDataRead as sd # Improt the program of SecondaryDataRead
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
class PaperModel(object): # Create the class of model.
    def __init__(self, is_training ,batch_size, num_steps, n_input, n_classes):# # The initlization function. is_training is to determination which the set is training_set，batch_size：the size of batch，num_steps：the number of steps，n_input：the number of input nodes of each step，n_classes：number of categories.
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.n_input = n_input
        self.n_classes = n_classes
        self.input_data = tf.placeholder(tf.float32, [None, num_steps, n_input])  # Definition the shape of input data: [None, num_steps, n_input].
        self.targets = tf.placeholder(tf.float32, [None, num_steps, n_classes])  # Definition the shape of output data: [None, num_steps, n_classes]
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0 # # If in the train: dropout_keep_prob=0.9，If in the test:dropout_keep_prob=1.0
        # Definition the cell of forward LSTM network.
        lstm_fw_cell = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=dropout_keep_prob)
            for _ in range(NUM_LAYERS)]
        # Definition the cell of backward LSTM network.
        lstm_bw_cell = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=dropout_keep_prob)
            for _ in range(NUM_LAYERS)]
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_fw_cell)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_bw_cell)
        self.input_dataone = tf.transpose(self.input_data, [1, 0, 2])
        self.input_dataone = tf.reshape(self.input_dataone, [-1, NUM_INPUT])
        self.input_dataone = tf.split(self.input_dataone,TRAIN_NUM_STEP)  # Transform the input data into the LSTM data.
        outputs, _, _ = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, self.input_dataone,dtype=tf.float32)  # outputs is Lstm output, the shape of outputs is [num_steps, batch_size, 2*HIDDEN]
        weightone = tf.get_variable("weightone", [2*HIDDEN_SIZE, HIDDEN_SIZE])
        biasone = tf.get_variable('biasone', [HIDDEN_SIZE])
        weighttwo = tf.get_variable("weighttwo", [HIDDEN_SIZE, HIDDEN_SIZE/2])
        biastwo = tf.get_variable('biastwo', [HIDDEN_SIZE/2])
        weightthree = tf.get_variable("weight", [HIDDEN_SIZE/2, N_CLASSES])
        biasthree = tf.get_variable('bias', [N_CLASSES])
        full_dropout_keep_prob = FULL_LAYER_DROPOUT if is_training else 1.0
        outputs = tf.reshape(outputs, [-1, 600])
        layer1 = tf.nn.relu(tf.matmul(outputs, weightone) + biasone)
        layer1 = tf.nn.dropout(layer1, full_dropout_keep_prob)
        layer2 = tf.nn.relu(tf.matmul(layer1, weighttwo) + biastwo)
        layer2 = tf.nn.dropout(layer2, full_dropout_keep_prob)
        self.logits = tf.nn.softmax(tf.matmul(layer2, weightthree) + biasthree) # Calculate prediction results.
def Test(session, test_x, test_y,model):
    logits = session.run([model.logits],feed_dict={model.input_data: test_x, model.targets: test_y})
    return logits
def TestTrain(session, test_x, test_y,model):
    logits = session.run([model.logits],feed_dict={model.input_data: test_x, model.targets: test_y})
    return logits
if __name__ == "__main__":
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("Paper_Model", reuse=None, initializer=initializer):
        test_model = PaperModel(False, TEST_BATCH_SIZE, TRAIN_NUM_STEP, NUM_INPUT, N_CLASSES)
    with tf.variable_scope("Paper_Model", reuse=True, initializer=initializer):
        testTrain_model = PaperModel(False, TESTTRAIN_BATCH_SIZE, TRAIN_NUM_STEP, NUM_INPUT, N_CLASSES)
    train_x, train_y, train_l, test_x, test_y, test_l,PriTestList,PriFileTestIndex = sd.MatureReadTrainDataBatch() # Get the train and test data.
    AccuracyTrain, AccurayTest = sd.Accuracy(train_l, test_l)
    AccuracyTrain = np.reshape(AccuracyTrain, [-1, 7])
    AccurayTest = np.reshape(AccurayTest, [-1, 7])
    saver = tf.train.Saver(max_to_keep=200)
    with tf.Session() as sess:
        saver.restore(sess, 'Model/First/model-50.ckpt') # Restore the record model
        print('sess, test_x, test_y, test_model', sess, test_x, test_y, test_model)
        Testlogits = Test(sess, test_x, test_y, test_model) # Get the prediction results of test data.
        Trainlogits = TestTrain(sess, train_x, train_y,  testTrain_model) # Get the prediction results of train data.
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
