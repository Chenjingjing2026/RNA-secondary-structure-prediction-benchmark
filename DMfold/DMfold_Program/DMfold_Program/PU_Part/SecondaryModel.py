import tensorflow as tf  # Improt the module of tensorflow
import numpy as np # Improt the module of numpy
import SecondaryDataRead as sd # Improt the program of SecondaryDataRead
import os
HIDDEN_SIZE = 300 # The number of LSTM hidden node.
NUM_LAYERS = 3  # The number of LSTM layer.
NUM_INPUT = 8   # The number of node in each step.
N_CLASSES = 7   # The number of the prediction result in each base.
TRAIN_BATCH_SIZE = 16   # train batch size
TEST_BATCH_SIZE = 316  # testing data size
TESTTRAIN_BATCH_SIZE = 2510 #training data size
TRAIN_NUM_STEP = 300    # the number of step in LSTM.
LSTM_KEEP_PROB = 0.9  # the rate of dropout in LSTM(encoder).
FULL_LAYER_DROPOUT = 0.9 # the rate of dropout in fully connected layer(decoder).
MAX_GRAD_NORM = 5       # Threshold to prevent gradient explosion.
LEARNing_rate_base = 0.002 # The learning rate
LEARNING_RATE_DECAY = 0.99 # Automatic decay rate of each epoch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 允许动态分配 GPU 内存
class PaperModel(object): # Create the class of model.
    def __init__(self, is_training ,batch_size, num_steps, n_input, n_classes):# The initlization function. is_training is to determination which the set is training_set，batch_size：the size of batch，num_steps：the number of steps，n_input：the number of input nodes of each step，n_classes：number of categories.
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.n_input = n_input
        self.n_classes = n_classes
        self.input_data = tf.placeholder(tf.float32, [None, num_steps, n_input]) # Definition the shape of input data: [None, num_steps, n_input].
        self.targets = tf.placeholder(tf.float32, [None, num_steps, n_classes]) # Definition the shape of output data: [None, num_steps, n_classes]
        self.length = tf.placeholder(tf.float32, [None]) # Definition the shape of output data: [None]
        self.Accuracy = tf.placeholder(tf.float32, [None, num_steps, n_classes]) # Definition the shape of accuracy: [None, num_steps, 2]
        self.Alllength = tf.placeholder(tf.float32) # Definition the average lengtth of the train or test data.
        AccurDemo = tf.reshape(self.Accuracy, [-1, 7]) # Transform the shape of Accuracy into [-1, 7].
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0 # If in the train: dropout_keep_prob=0.9，If in the test:dropout_keep_prob=1.0
        # Definition the cell of forward LSTM network.
        lstm_fw_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=dropout_keep_prob) for _ in range(NUM_LAYERS)]
        # Definition the cell of backward LSTM network.
        lstm_bw_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=dropout_keep_prob) for _ in range(NUM_LAYERS)]
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_fw_cell)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_bw_cell)
        self.input_dataone = tf.transpose(self.input_data, [1, 0, 2])
        self.input_dataone = tf.reshape(self.input_dataone, [-1, NUM_INPUT])
        self.input_dataone = tf.split(self.input_dataone, TRAIN_NUM_STEP) # Transform the input data into the LSTM data.
        outputs, _, _ = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, self.input_dataone, dtype=tf.float32)  # outputs is Lstm output, the shape of outputs is [num_steps, batch_size, 2*HIDDEN]
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, 2*HIDDEN_SIZE]) # First, transform the shape of outputs into [batch, 2*HIDDEN*num_steps]，Secondary transform the shape into [batch*num_steps, 2*HIDDEN]格式
        weightone = tf.get_variable("weightone", [2*HIDDEN_SIZE, HIDDEN_SIZE])
        biasone = tf.get_variable('biasone', [HIDDEN_SIZE])
        weighttwo = tf.get_variable("weighttwo", [HIDDEN_SIZE, HIDDEN_SIZE/2])
        biastwo = tf.get_variable('biastwo', [HIDDEN_SIZE/2])
        weightthree = tf.get_variable("weight", [HIDDEN_SIZE/2, N_CLASSES])
        biasthree = tf.get_variable('bias', [N_CLASSES])
        full_dropout_keep_prob = FULL_LAYER_DROPOUT if is_training else 1.0 # When in training full_dropout_keep_prob = 0.9, else full_dropout_keep_prob = 1.0
        # Forward propagation of neural network.
        layer1 = tf.nn.relu(tf.matmul(outputs, weightone) + biasone)
        layer1 = tf.nn.dropout(layer1, full_dropout_keep_prob)
        layer2 = tf.nn.relu(tf.matmul(layer1, weighttwo) + biastwo)
        layer2 = tf.nn.dropout(layer2, full_dropout_keep_prob)
        logits = tf.nn.relu(tf.matmul(layer2, weightthree) + biasthree)  # Calculate prediction results.
        correect_pred = tf.equal(tf.argmax(logits*AccurDemo, 1), tf.argmax(tf.reshape(self.targets, [-1, 7])*AccurDemo, 1)) # Calculate the accuracy of prediction, logits*AccurDemo is used to prevent the adverse effects of padding zero vectors.
        self.accuracy = tf.reduce_mean(tf.cast(correect_pred, tf.float32)) # Calculate the average accuracy.
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1, 7]), logits=logits) # Calculate the loss.
        label_weights = tf.sequence_mask(self.length, maxlen=300, dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1]) # transform the label_weights into One-dimensional matrix.
        self.cost = tf.reduce_mean(loss*label_weights)*(300/self.Alllength) # Calculate the average loss.
        if not is_training: return
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM) # Prevent gradient explosion
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNing_rate_base, global_step, 100, LEARNING_RATE_DECAY, staircase=True) # Learning rate is automatically decay.
        # Neural network back propagation
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step) # Train the model
def Run_epoch(session, train_x_batch, train_y_batch, train_l,AccuracyTrain,model):
    for i in range(100): # 100 times of per epoch.
        id = np.random.choice(TESTTRAIN_BATCH_SIZE, size=TRAIN_BATCH_SIZE, replace=False)
        train_batch_x = train_x_batch[id]
        train_batch_y = train_y_batch[id]
        train_batch_l = train_l[id]   # Randomly select TRAIN_BATCH_SIZE data
        AccuracyTrain_batch = AccuracyTrain[id]
        Tcount =sd.Alllength(train_batch_l) # Get the average length of train data.
        cost, accuracy,  _ = session.run([model.cost, model.accuracy, model.train_op], feed_dict={model.input_data: train_batch_x, model.targets: train_batch_y, model.length: train_batch_l,model.Accuracy:AccuracyTrain_batch, model.Alllength:Tcount})
        if(i%10 ==0): # output the training accuracy.
            print(Tcount)
            print("After %d steps, cost is %.3f" % (i, cost))
            print("After %d steps, accuracy is %.3f" % (i,(Tcount-(300-accuracy*300))/Tcount))
'''
Test accuracy in the test set after each epoch of training
'''
def Test(session, test_x, test_y, test_l, AccuracyTest, model):
    Tcount = sd.Alllength(test_l)
    Testcost, accuracy = session.run([model.cost, model.accuracy], feed_dict={model.input_data: test_x, model.targets: test_y, model.length: test_l,model.Accuracy: AccuracyTest, model.Alllength: Tcount})
    print(Tcount)
    print("After %d test, cost is %.3f" % (1, Testcost))
    print("After %d accuracy, cost is %.3f" % (1, (Tcount - (300 - accuracy * 300)) / Tcount))
    return (Tcount - (300 - accuracy * 300)) / Tcount
'''
Test accuracy using all training data after each epoch.
'''
def TestTrain(session, test_x, test_y, test_l, AccuracyTest, model):
    Tcount = sd.Alllength(test_l)
    Testcost, accuracy = session.run([model.cost, model.accuracy], feed_dict={model.input_data: test_x, model.targets: test_y,model.length: test_l, model.Accuracy: AccuracyTest,model.Alllength: Tcount})
    print(Tcount)
    print(len(test_x))
    print("After %d test, cost is %.3f" % (1, Testcost))
    print("After %d accuracy, Traintest accuracy is %.3f" % (1, (Tcount - (300 - accuracy * 300)) / Tcount))
if __name__ == "__main__":

    initializer = tf.random_uniform_initializer(-0.05, 0.05)    # Initialization the weights.
    with tf.variable_scope("Paper_Model", reuse=None, initializer=initializer):
        train_model = PaperModel(True,TRAIN_BATCH_SIZE, TRAIN_NUM_STEP, NUM_INPUT, N_CLASSES)
    with tf.variable_scope("Paper_Model", reuse=True, initializer=initializer):
        test_model = PaperModel(False, TEST_BATCH_SIZE, TRAIN_NUM_STEP, NUM_INPUT, N_CLASSES)
    with tf.variable_scope("Paper_Model", reuse=True, initializer=initializer):
        testTrain_model = PaperModel(False, TESTTRAIN_BATCH_SIZE, TRAIN_NUM_STEP, NUM_INPUT, N_CLASSES)
    train_x, train_y, train_l, test_x, test_y, test_l,PriTestList,PriFileTestIndex= sd.MatureReadTrainDataBatch() # Get the train and test data.

    AccuracyTrain, AccurayTest = sd.Accuracy(train_l, test_l)
    saver = tf.train.Saver(max_to_keep=200)
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        count = 0
        index = 0
        for i in range(200):
            print("******************%d********************"%i)
            Run_epoch(sess, train_x, train_y, train_l, AccuracyTrain,train_model) # train model.
            TestAccuracy = Test(sess, test_x, test_y, test_l, AccurayTest,test_model)   # test model using test set.
            TestTrain(sess, train_x, train_y, train_l, AccuracyTrain, testTrain_model)  # test model using train set.
            Flag = False
            if(TestAccuracy > count):
                Flag = True
                count = TestAccuracy
                index = i
            if(TestAccuracy>=0.85 and Flag == True): # When the testing accuracy more than 0.85, record the model.
                saver.save(sess, '/home/chenjingjing/Models/DMfold/Model/bprna_1m/model.ckpt', global_step=i)

        saver.save(sess, '/home/chenjingjing/Models/DMfold/Model/bprna_1m/model.ckpt', global_step=i)
    print("Finish")

