class PaperModel(tf.Module):  # Inherit from tf.Module


    def __init__(self, is_training ,batch_size, num_steps, n_input, n_classes):# # The initlization function. is_training is to determination which the set is training_set，batch_size：the size of batch，num_steps：the number of steps，n_input：the number of input nodes of each step，n_classes：number of categories.
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.n_input = n_input
        self.n_classes = n_classes
        # Define the model's layers
        self.dense = tf.keras.layers.Dense(n_classes)

        # Use tf.keras.Input to define the shape of input data
        self.input_data = tf.keras.Input(shape=(num_steps, n_input))
        self.targets = tf.keras.Input(shape=(num_steps, n_classes))


        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0 # # If in the train: dropout_keep_prob=0.9，If in the test:dropout_keep_prob=1.0

        # 定义前向和后向 LSTM 单元
        lstm_fw_cells = [
            tf.keras.layers.LSTMCell(units=HIDDEN_SIZE, dropout=1 - dropout_keep_prob)
            for _ in range(NUM_LAYERS)
        ]
        lstm_bw_cells = [
            tf.keras.layers.LSTMCell(units=HIDDEN_SIZE, dropout=1 - dropout_keep_prob)
            for _ in range(NUM_LAYERS)
        ]

        # 使用 MultiRNNCell 堆叠多层 LSTM
        fw_cell = tf.keras.layers.StackedRNNCells(lstm_fw_cells)
        bw_cell = tf.keras.layers.StackedRNNCells(lstm_bw_cells)

        self.input_dataone = tf.transpose(self.input_data, [1, 0, 2])
        self.input_dataone = tf.reshape(self.input_dataone, [-1, NUM_INPUT])
        self.input_dataone = tf.split(self.input_dataone,TRAIN_NUM_STEP)  # Transform the input data into the LSTM data.


        # 使用 Bidirectional 包装器
        bidirectional_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(fw_cell, return_sequences=True, return_state=True),
            backward_layer=tf.keras.layers.RNN(bw_cell, return_sequences=True, return_state=True, go_backwards=True)
            # 设置 go_backwards=True
        )

        # 调用 Bidirectional RNN
        outputs, *states = bidirectional_rnn(self.input_data)


        # 权重和偏置
        weightone = tf.Variable(tf.random.normal([2 * HIDDEN_SIZE, HIDDEN_SIZE]), name="weightone")
        biasone = tf.Variable(tf.zeros([HIDDEN_SIZE]), name="biasone")

        weighttwo = tf.Variable(tf.random.normal([HIDDEN_SIZE, HIDDEN_SIZE // 2]), name="weighttwo")
        biastwo = tf.Variable(tf.zeros([HIDDEN_SIZE // 2]), name="biastwo")

        weightthree = tf.Variable(tf.random.normal([HIDDEN_SIZE // 2, N_CLASSES]), name="weightthree")
        biasthree = tf.Variable(tf.zeros([N_CLASSES]), name="biasthree")


        full_dropout_keep_prob = FULL_LAYER_DROPOUT if is_training else 1.0
        layer1 = tf.nn.relu(tf.matmul(outputs, weightone) + biasone)
        layer1 = tf.nn.dropout(layer1, full_dropout_keep_prob)
        layer2 = tf.nn.relu(tf.matmul(layer1, weighttwo) + biastwo)
        layer2 = tf.nn.dropout(layer2, full_dropout_keep_prob)
        self.logits = tf.nn.softmax(tf.matmul(layer2, weightthree) + biasthree) # Calculate prediction results.