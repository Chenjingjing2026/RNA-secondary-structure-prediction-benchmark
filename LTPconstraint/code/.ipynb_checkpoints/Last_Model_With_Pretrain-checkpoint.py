#kears == 2.3.1
import os
from keras.layers import Lambda
import tensorflow as tf
from tensorflow.keras import layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')
# from sklearn.model_selection import train_test_split

# print('Get GPU')
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# print('Get GPU Finish')
# # # tf.config.experimental.set_visible_devices(devices=gpus[0:3], device_type='GPU')
# for gpu in gpus:
#     print('set',gpu,'memory_growth')
#     tf.config.experimental.set_memory_growth(gpu, True)
# print('set gpu memory_growth Finish')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
# [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11178)])
#
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[1],
# [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11111)])



import keras

from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import math
from keras.layers import Dropout, Input, Dense, Activation, BatchNormalization, Conv2D, Embedding, LSTM
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.initializers import Constant, RandomUniform
from keras.backend import softsign, sigmoid, relu, sign
from keras.utils import to_categorical
from scipy.sparse import diags
from keras.layers.wrappers import Bidirectional
from keras.utils import multi_gpu_model
from My_Layers import PositionEncoding, ScaledDotProductAttention, MultiHeadAttention, PositionWiseFeedForward, LayerNormalization, ConstraintAndResult, downsample, upsample, Generator, constraint_matrix_batch, F1_loss, logistic_regression_loss, auc, weighted_cross_entropy_with_logits_Me, pr_auc, val_test
from tensorflow.keras.optimizers import Adam
from keras.utils.data_utils import Sequence
print('import Finish')

#added
pretrain_modelpath = '/home/chenjingjing/Models/LTPConstraint/model/512/pretrain_RNAStrAlign.h5'
train_modelpath = '/home/chenjingjing/Models/LTPConstraint/model/512/train_RNAStrAlign.h5'

# 獲取文件夾路徑
pretrain_dir = os.path.dirname(pretrain_modelpath)
train_dir = os.path.dirname(train_modelpath)

# 檢查並建立文件夾
if not os.path.exists(pretrain_dir):
    os.makedirs(pretrain_dir)
    print(f"文件夾 {pretrain_dir} 已建立")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print(f"文件夾 {train_dir} 已建立")


# 定义策略
# print('定义策略')
# strategy = tf.distribute.MirroredStrategy()
# print('定义策略Finish')
# rate is the rate of test set
class MnistSequence(Sequence):
    def __init__(self, x_set, m_set, y_set, batch_size, rate, is_train=True):

        self.x, self.y, self.m = x_set, y_set, m_set
        self.batch_size = batch_size
        self.rate = rate
        self.is_train = is_train

    def __len__(self):
        if self.is_train:
            lens = int(np.ceil(len(self.x)*(1-self.rate) / float(self.batch_size)))
        else:
            lens = int(np.ceil(len(self.x)*self.rate / float(self.batch_size)))
        return lens


    def __getitem__(self, idx):
        if self.is_train:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_m = self.m[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            lens = int(np.ceil(len(self.x)*(1-self.rate)))
            batch_x = self.x[lens + idx * self.batch_size:lens + (idx + 1) * self.batch_size]
            batch_m = self.m[lens + idx * self.batch_size:lens + (idx + 1) * self.batch_size]
            batch_y = self.y[lens + idx * self.batch_size:lens + (idx + 1) * self.batch_size]

        return (batch_x, batch_m), batch_y

class PretrainSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, rate, is_train=True):

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.rate = rate
        self.is_train = is_train

    def __len__(self):
        if self.is_train:
            lens = int(np.ceil(len(self.x)*(1-self.rate) / float(self.batch_size)))
        else:
            lens = int(np.ceil(len(self.x)*self.rate / float(self.batch_size)))
        return lens


    def __getitem__(self, idx):
        if self.is_train:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            lens = int(np.ceil(len(self.x)*(1-self.rate)))
            batch_x = self.x[lens + idx * self.batch_size:lens + (idx + 1) * self.batch_size]
            batch_y = self.y[lens + idx * self.batch_size:lens + (idx + 1) * self.batch_size]

        return batch_x, batch_y

def split_train_test(seq_list, M, label_atrix, rate):
    lens = int(np.ceil(len(seq_list) * (1-rate)))
    X_train = seq_list[0:lens]
    M_train = M[0:lens]
    Y_train = label_atrix[0:lens]
    X_test = seq_list[lens:len(seq_list)]
    Y_test = label_atrix[lens:len(seq_list)]
    M_test = M[lens:len(seq_list)]
    return X_train, M_train, Y_train, X_test, M_test, Y_test


# SHAPE_Model = keras.models.load_model('../model/SHAPE_Model.h5',custom_objects={
#     'PositionEncoding': PositionEncoding,
#     'MultiHeadAttention': MultiHeadAttention,
#     'PositionWiseFeedForward': PositionWiseFeedForward,
#     'LayerNormalization': LayerNormalization
#
#                                                                                 })
# result = SHAPE_Model.predict(seq_list, batch_size=128)
# for i in range(result.shape[0]):
#     for j in range(result.shape[1]):
#         if result[i][j] <= 0.5:
#             result[i][j] = 2 * (0.3603 - 0.6624) * result[i][j] + 0.6624
#         else:
#             result[i][j] = 2 * (0.214 - 0.3603) * (result[i][j] - 1) + 0.214
#         result[i][j] = 1.7674 * (math.exp(0.6624 - result[i][j]) - 1)
# result_T = result[:, :, np.newaxis]
# result = result[:, np.newaxis, :]
# final_result = []
# for i in range(result.shape[0]):
#     final_result.append(np.dot(result_T[i], result[i]))
# final_result = np.array(final_result)
    # # 将 TensorFlow 操作包装在 Lambda 层中
def symmetric_mean(x):
    x = tf.squeeze(x, -1)
    return (x + tf.transpose(x, (0, 2, 1))) / 2
def pretrain_model_get(limit_legnth, vocab_size, max_seq_len, model_dim, dropout_rate, encoder_stack, n_heads, feed_forward_size):
    encoder_inputs = Input(shape=(max_seq_len,), name='encoder_inputs')
    if K.dtype(encoder_inputs) != 'int32':
        inputs = K.cast(encoder_inputs, 'int32')
    masks = K.equal(encoder_inputs, 0)
    embeddings = Embedding(input_dim=6, output_dim=model_dim, input_length=max_seq_len)(encoder_inputs)
    position_encodings = PositionEncoding(model_dim)(embeddings)
    encodings = Bidirectional(LSTM(model_dim//2, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, kernel_initializer='RandomNormal', dropout= 0.3, recurrent_initializer='RandomNormal', bias_initializer='zero'))(embeddings)
    encodings = Lambda(lambda x: K.dropout(x, dropout_rate))(encodings)
    for i in range(encoder_stack):
        # attention = MultiHeadAttention(n_heads, model_dim // n_heads)
        # attention_input = [encodings, encodings, encodings, masks]
        attention_out = Lambda(lambda x: MultiHeadAttention(n_heads, model_dim // n_heads)([x, x, x, masks]))(encodings)
        attention_out = Lambda(lambda x: x + encodings)(attention_out)
        attention_out = LayerNormalization()(attention_out)
        ff_out = Lambda(lambda x: PositionWiseFeedForward(model_dim, feed_forward_size)(x))(attention_out)
        # encodings = LayerNormalization()(ff_out)
        encodings = Lambda(lambda x: LayerNormalization()(x))(ff_out)

    #lamba
    encodings = Lambda(lambda x: x[0] + x[1])([encodings, position_encodings])
    encode_pair = Lambda(lambda x: tf.expand_dims(x, 1))(encodings)
    encode_pair = Lambda(lambda x: K.repeat_elements(x, inputs.shape[1], 1))(encode_pair)
    encode_pair = Lambda(lambda x: tf.concat([tf.transpose(x, perm=[0, 2, 1, 3]), x], 3))(encode_pair)
    outputs = Lambda(lambda x: Generator(x, max_seq_len, model_dim * 2, 1))(encode_pair)
    outputs = Lambda(symmetric_mean)(outputs)

    pretrain_model = Model(inputs=encoder_inputs, outputs=outputs, name="pretrain_model")

    return pretrain_model

def pretrain_model_get123(limit_legnth, vocab_size, max_seq_len, model_dim, dropout_rate, encoder_stack, n_heads, feed_forward_size):
    encoder_inputs = Input(shape=(max_seq_len,), name='encoder_inputs')
    # SHAPE_inputs = Input(shape=(max_seq_len, max_seq_len), name='SHAPE_inputs')
    if K.dtype(encoder_inputs) != 'int32':
        inputs = K.cast(encoder_inputs, 'int32')
    masks = K.equal(encoder_inputs, 0)
    embeddings = Embedding(input_dim=6, output_dim=model_dim, input_length=max_seq_len)(encoder_inputs)
    position_encodings = PositionEncoding(model_dim)(embeddings)
    # Embedings + Postion-encodings
    # encodings = embeddings + position_encodings
    # recurrent_activation='sigmoid' added then it can be accelerate by cudnn
    encodings = Bidirectional(LSTM(model_dim//2, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, kernel_initializer='RandomNormal', dropout= 0.3, recurrent_initializer='RandomNormal', bias_initializer='zero'))(embeddings)
    # Dropout
    encodings = K.dropout(encodings, dropout_rate)
    # 多个encoder串联，上一个的输出是下一个encoder的输入
    for i in range(encoder_stack):
        # Multi-head-Attention
        attention = MultiHeadAttention(n_heads, model_dim // n_heads)
        attention_input = [encodings, encodings, encodings, masks]
        attention_out = attention(attention_input)
        # Add & Norm
        attention_out += encodings
        attention_out = LayerNormalization()(attention_out)
        # Feed-Forward
        ff = PositionWiseFeedForward(model_dim, feed_forward_size)
        ff_out = ff(attention_out)
        # Add & Norm
        encodings = LayerNormalization()(ff_out)
    # encodings = encodings + position_encodings
    # print('encodings',encodings)
    # #变换为二维矩阵
    # encode_pair = tf.expand_dims(encodings, 1)
    # encode_pair = K.repeat_elements(encode_pair, inputs.shape[1], 1)
    # encode_pair = tf.concat([tf.transpose(encode_pair, perm=[0, 2, 1, 3]), encode_pair], 3)

    #lamba
    encodings = Lambda(lambda x: x[0] + x[1])([encodings, position_encodings])
    encode_pair = Lambda(lambda x: tf.expand_dims(x, 1))(encodings)
    encode_pair = Lambda(lambda x: K.repeat_elements(x, inputs.shape[1], 1))(encode_pair)
    encode_pair = Lambda(lambda x: tf.concat([tf.transpose(x, perm=[0, 2, 1, 3]), x], 3))(encode_pair)



    # encode_pair = Lambda(lambda x: tf.expand_dims(x, 1))(encodings)
    # encode_pair = Lambda(lambda x: K.repeat_elements(x, inputs.shape[1], 1))(encode_pair)
    # encode_pair = Lambda(lambda x: tf.concat([tf.transpose(x, perm=[0, 2, 1, 3]), x], 3))(encode_pair)
    # 放入generator
    outputs = Lambda(lambda x: Generator(x, max_seq_len, model_dim * 2, 1))(encode_pair)

    print('Generator outputs',outputs.shape)

    print('squeeze outputs',outputs.shape)
    # outputs = Lambda(lambda x: tf.reduce_mean(x, axis=-1))(encode_pair)

    # outputs = (outputs + tf.transpose(outputs, (0, 2, 1))) / 2



    outputs = Lambda(symmetric_mean)(outputs)
    # outputs = (outputs + tf.transpose(outputs, (0, 2, 1))) / 2
    print(f"embeddings shape: {embeddings.shape}")
    print(f"attention_out shape: {attention_out.shape}")
    print(f"encode_pair shape: {encode_pair.shape}")
    print(f"outputs shape: {outputs.shape}")

    # 构建模型
    print('encoder_inputs',encoder_inputs.shape)
    print('outputs',outputs.shape)

    pretrain_model = Model(inputs=encoder_inputs, outputs=outputs, name="pretrain_model")

    #
    #
    # pretrain_model = Model(inputs=encoder_inputs, outputs=outputs, name="pretrain_model")
    return pretrain_model



def nopre_train(seq_list, label_atrix, batch_size, epochs):
    limit_length = 128
    vocab_size = 5
    max_seq_len = limit_length
    model_dim = 10
    dropout_rate = 0.2
    encoder_stack = 6
    n_heads = 2
    feed_forward_size = 1024
    steps = 16

    pretrain_model = pretrain_model_get(limit_legnth=limit_length,
                               vocab_size=vocab_size,
                               max_seq_len=limit_length,
                               model_dim=model_dim,
                               dropout_rate=dropout_rate,
                               encoder_stack=encoder_stack,
                               n_heads=n_heads,
                               feed_forward_size=feed_forward_size
                               )
    encoder_inputs = pretrain_model.input
    x = pretrain_model.output
    constraint_inputs = Input(shape=(max_seq_len, max_seq_len), name='constraint_inputs')
    # 添加约束反推结果矩阵
    CAR_layer = ConstraintAndResult(max_seq_len, steps)
    pred = CAR_layer(x, constraint_inputs)

    model = Model(inputs=(encoder_inputs, constraint_inputs), outputs=pred, name="sequence_encoder_model")
    model.summary()

    # model.compile(optimizer=Adam(lr=2e-4, beta_1=0.95, beta_2=0.98, epsilon=1e-7),
    #               loss=F1_loss, metrics=['binary_accuracy'])
    model.compile(optimizer='Adam',
                  loss=F1_loss, metrics=['binary_accuracy'])

    M = constraint_matrix_batch(seq_list)
    # td = TensorBoard(log_dir='../logs/nopretrain_train_RfamUn80', histogram_freq=0, write_graph=True, write_images=True)
    rp = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=3,
        verbose=0,
        mode='auto',
        epsilon=4e-4,
        min_lr=1e-11
    )
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=0,
        mode='auto'
    )
    per = np.random.permutation(seq_list.shape[0])
    seq_list = seq_list[per, :]
    M = M[per, :, :]
    label_atrix = label_atrix[per, :, :]

    model.fit_generator(MnistSequence(seq_list, M, label_atrix, batch_size, 0.2),
                        validation_data=MnistSequence(seq_list, M, label_atrix, batch_size, 0.2, False), epochs=epochs,
                        callbacks=[rp, es], workers=2)
    model.save_weights(modelpath)


def pretrain_Model(seq_list, label_atrix, batch_size, epochs):
    limit_length = 512
    vocab_size = 5
    max_seq_len = limit_length
    model_dim = 10
    dropout_rate = 0.2
    encoder_stack = 6
    n_heads = 2
    # test save

    # test
    feed_forward_size = 1024
    model = pretrain_model_get(limit_legnth=limit_length,
                               vocab_size=vocab_size,
                               max_seq_len=limit_length,
                               model_dim=model_dim,
                               dropout_rate=dropout_rate,
                               encoder_stack=encoder_stack,
                               n_heads=n_heads,
                               feed_forward_size=feed_forward_size
                               )
    #check model
    # model.summary()

    # 把模型转到2块gpu上
    # model = multi_gpu_model(model, gpus = 2)

    # Adamoptimizer = Adam(lr=2e-4, beta_1=0.95, beta_2=0.98, epsilon=1e-7)
    Adamoptimizer = 'Adam'
    # model.compile(optimizer=Adamoptimizer,loss=weighted_cross_entropy_with_logits_Me, metrics=['binary_accuracy'])

    model.compile(optimizer=Adamoptimizer, loss=weighted_cross_entropy_with_logits_Me, metrics=['binary_accuracy'])
    return model

def pretrain(seq_list, label_atrix, batch_size, epochs):
    print('pretrain')
    #for test
    seq_list = seq_list[0:100]
    label_atrix = label_atrix[0:100]
    print('len seq_list', len(seq_list))
    print('len label_atrix', len(label_atrix))

    model = pretrain_Model(seq_list, label_atrix, batch_size, epochs)
    print("Model Training ... ")
    td = TensorBoard(log_dir='../logs/pretrain_Rfam512_wcre', histogram_freq=0, write_graph=True, write_images=True)
    rp = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=3,
        verbose=0,
        mode='auto',
        epsilon=4e-4,
        min_lr=1e-11
    )
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=0,
        mode='auto'
    )
    per = np.random.permutation(seq_list.shape[0])
    seq_list = seq_list[per, :]
    label_atrix = label_atrix[per, :, :]
    print("Model fit_generator ... ")
    # model.fit(seq_list, label_atrix, batch_size=batch_size, epochs=epochs, validation_split=0.2, workers = 3, callbacks=[td, rp, es])
    model.fit_generator(PretrainSequence(seq_list, label_atrix, batch_size, 0.2),
                        validation_data=PretrainSequence(seq_list, label_atrix, batch_size, 0.2, False), epochs=epochs,
                        callbacks=[td, rp, es], workers=1,use_multiprocessing=False)
    print(pretrain_modelpath)
    model.save_weights(pretrain_modelpath)


def Train_Model(pretrain_model):

    max_seq_len = 128
    steps = 16
    encoder_inputs = pretrain_model.input
    pretrain_model_output = pretrain_model.output

    constraint_inputs = Input(shape=(max_seq_len, max_seq_len), name='constraint_inputs')
    # 添加约束反推结果矩阵
    CAR_layer = ConstraintAndResult(max_seq_len, steps)
    print('CAR_layer')
    print('pretrain_model_output',pretrain_model_output)
    print('constraint_inputs', constraint_inputs)
    pred = CAR_layer(pretrain_model_output,constraint_inputs)
    print('pred')
    model = Model(inputs=(encoder_inputs, constraint_inputs), outputs=pred, name="sequence_encoder_model")
    # model.summary()
    # parallel_model = multi_gpu_model(model, gpus = 2)
    # change to Adamoptimizer = 'Adam'
    #     model.compile(optimizer=Adam(lr=2e-4, beta_1=0.95, beta_2=0.98, epsilon=1e-7), loss=F1_loss, metrics=['binary_accuracy'])
    model.compile(optimizer='Adam', loss=F1_loss,
                  metrics=['binary_accuracy'])

def start_train(seq_list, label_atrix, batch_size, epochs):
    # for test
    seq_list = seq_list[0:100]
    label_atrix = label_atrix[0:100]
    print('len seq_list', len(seq_list))
    print('len label_atrix', len(label_atrix))
    print('Strat_train')
    M = constraint_matrix_batch(seq_list)
    # model_name = '../model/128/pretrain_Rfam_wcreloss.h5'

    print('read pretrain_model')
    pretrain_model = pretrain_Model(seq_list, label_atrix, batch_size, epochs)
    print('load pretrain_model')
    pretrain_model.load_weights(pretrain_modelpath)

    print('Start Train init success')


    # pretrain_model = keras.models.load_model(model_name, custom_objects={
    #     'PositionEncoding': PositionEncoding,
    #     'MultiHeadAttention': MultiHeadAttention,
    #     'PositionWiseFeedForward': PositionWiseFeedForward,
    #     'LayerNormalization': LayerNormalization,
    #     'F1_loss': F1_loss,
    #     'weighted_cross_entropy_with_logits_Me': weighted_cross_entropy_with_logits_Me,
    #     'logistic_regression_loss':logistic_regression_loss,
    #     'auc': auc,
    #     'pr_auc': pr_auc
    # })

    # for layer in pretrain_model.layers:
    #     layer.trainable = False  # 原来的不训练

    model = Train_Model(pretrain_model)

    # td = TensorBoard(log_dir='../logs/train_tmRNA_pretrainRfam512', histogram_freq=0, write_graph=True, write_images=True)
    rp = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=3,
        verbose=0,
        mode='auto',
        epsilon=4e-4,
        min_lr=1e-11
    )
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=0,
        mode='auto'
    )
    per = np.random.permutation(seq_list.shape[0])
    seq_list = seq_list[per, :]
    M = M[per, :, :]
    label_atrix = label_atrix[per, :, :]

    # X_train, M_train, Y_train, X_test, M_test, Y_test = split_train_test(seq_list, M, label_atrix, 0.2)
    model.fit_generator(MnistSequence(seq_list, M, label_atrix, batch_size, 0.2), validation_data=MnistSequence(seq_list, M, label_atrix, batch_size, 0.2, False),epochs=epochs, callbacks=[rp, es], workers=2)
    # model.fit((seq_list, M), label_atrix, workers= 2, batch_size=batch_size, epochs=epochs, validation_split=0.2,
    #       callbacks=[td, rp, es])
    model.save_weights(train_modelpath)




def continue_pretrain(seq_list, label_atrix, batch_size, epochs):
    model_name = pretrain_modelpath
    model = keras.models.load_model(model_name, custom_objects={
        'PositionEncoding': PositionEncoding,
        'MultiHeadAttention': MultiHeadAttention,
        'PositionWiseFeedForward': PositionWiseFeedForward,
        'LayerNormalization': LayerNormalization,
        'F1_loss': F1_loss,
        'weighted_cross_entropy_with_logits_Me': weighted_cross_entropy_with_logits_Me,
        'logistic_regression_loss': logistic_regression_loss,
        'auc': auc,
        'pr_auc': pr_auc
    })
    # model.compile(optimizer=Adam(lr=2e-4, beta_1=0.95, beta_2=0.98, epsilon=1e-7),
    #               loss=weighted_cross_entropy_with_logits_Me, metrics=['binary_accuracy'])
    model.compile(optimizer='Adam',
                  loss=weighted_cross_entropy_with_logits_Me, metrics=['binary_accuracy'])
    print("Model Training ... ")
    # td = TensorBoard(log_dir='../logs/pretrain_tmRNA_WithPreRfam512', histogram_freq=0, write_graph=True, write_images=True)
    rp = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=3,
        verbose=0,
        mode='auto',
        epsilon=4e-4,
        min_lr=1e-11
    )
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=0,
        mode='auto'
    )
    per = np.random.permutation(seq_list.shape[0])
    seq_list = seq_list[per, :]
    label_atrix = label_atrix[per, :, :]

    model.fit_generator(PretrainSequence(seq_list, label_atrix, batch_size, 0.2),
                        validation_data=PretrainSequence(seq_list, label_atrix, batch_size, 0.2, False), epochs=epochs,
                        callbacks=[rp, es], workers=1)
    # model.fit(seq_list, label_atrix, batch_size=batch_size, epochs=epochs, validation_split=0.2,
    #           callbacks=[td, rp, es])
    model.save(pretrain_modelpath)





def continue_train(model_name, seq_list, label_atrix, batch_size, epochs):

    model = keras.models.load_model(model_name, custom_objects={
        'PositionEncoding': PositionEncoding,
        'MultiHeadAttention': MultiHeadAttention,
        'PositionWiseFeedForward': PositionWiseFeedForward,
        'LayerNormalization': LayerNormalization,
        'ConstraintAndResult': ConstraintAndResult,
        'F1_loss': F1_loss
    })
    # model.compile(optimizer=Adam(lr=2e-4, beta_1=0.95, beta_2=0.98, epsilon=1e-7),
    #               loss=F1_loss, metrics=['binary_accuracy'])
    model.compile(optimizer='Adam',
                  loss=F1_loss, metrics=['binary_accuracy'])

    M = constraint_matrix_batch(seq_list)
    td = TensorBoard(log_dir='../logs/Last_model', histogram_freq=0, write_graph=True, write_images=True)
    rp = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=3,
        verbose=0,
        mode='auto',
        epsilon=4e-4,
        min_lr=1e-11
    )
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=60,
        verbose=0,
        mode='auto'
    )
    model.fit((seq_list, M), label_atrix, batch_size=batch_size, epochs=epochs, validation_split=0.2,
              callbacks=[td, rp, es])
    model.save(train_modelpath)

def start_predict(model_name, start_rate, end_rate, seq_list, label_atrix, batch_size):
    print('read pretrain_model')
    pretrain_model = pretrain_Model(seq_list, label_atrix, batch_size, epochs)
    print('load pretrain_model')
    pretrain_model.load_weights(pretrain_modelpath)

    print('read train_model')
    model = Train_Model(pretrain_model)
    print('load train_model')
    model.load_weights(train_modelpath)

    # model = keras.models.load_model(model_name, custom_objects={
    #     'PositionEncoding': PositionEncoding,
    #     'MultiHeadAttention': MultiHeadAttention,
    #     'PositionWiseFeedForward': PositionWiseFeedForward,
    #     'LayerNormalization': LayerNormalization,
    #     'ConstraintAndResult': ConstraintAndResult,
    #     'F1_loss': F1_loss,
    #     'weighted_cross_entropy_with_logits_Me': weighted_cross_entropy_with_logits_Me,
    #     'logistic_regression_loss': logistic_regression_loss,
    #     'auc': auc,
    #     'pr_auc': pr_auc
    # })
    length = int(seq_list.shape[0])
    start = int(np.ceil(start_rate * length))
    end = int(np.ceil(end_rate * length))
    seq_list = seq_list[start:end]
    label_atrix = label_atrix[start:end]
    M = constraint_matrix_batch(seq_list)
    prediction = model.predict((seq_list, M), batch_size=batch_size)
    P, R, cell = val_test(label_atrix, prediction)
    F1 = (-1)*F1_loss(label_atrix, prediction)
    print(P.numpy(), R.numpy(), F1.numpy(), cell[0].numpy(), cell[1].numpy(), cell[2].numpy(), cell[3].numpy())



# outputs = ShapeOutputslayer(max_seq_len)((outputs, SHAPE_inputs))
# outputs = sigmoid(Foutputs * SHAPE_inputs)


Save_dir = "/home/chenjingjing/Models/LTPConstraint/data/512/RNAStrAlign/"
# Save_dir = '/home/chenjingjing/Models/LTPConstraint/data/512/'
# data = np.load(Save_dir+'data_telomerase.npz')
data = np.load(Save_dir+'RNAStrAlign_512.npz')
seq_list = data['arr_0']
label_atrix = data['arr_1']
# SHAPE_LABEL = data['arr_2']
batch_size = 100
epochs = 1

#Full running step
# pretrain(seq_list, label_atrix, batch_size, epochs)
# start_train(seq_list[0:int(0.9*len(seq_list))], label_atrix[0:int(0.9*len(seq_list))], batch_size, epochs)
# start_predict(model_name, 0.9, 1.0, seq_list, label_atrix, batch_size)

#Test Step
pretrain(seq_list, label_atrix, batch_size, epochs) #Fininsh
start_train(seq_list[0:int(0.9*len(seq_list))], label_atrix[0:int(0.9*len(seq_list))], batch_size, epochs)
start_predict(model_name, 0.9, 1.0, seq_list, label_atrix, batch_size)

# pretrain(seq_list, label_atrix, batch_size, epochs)
# # continue_pretrain(seq_list, label_atrix, batch_size, epochs)
# start_train(seq_list[0:int(0.9*len(seq_list))], label_atrix[0:int(0.9*len(seq_list))], batch_size, epochs)
# # # nopre_train(seq_list, label_atrix, batch_size, epochs)
# # model_name = '../model/Last_model.h5'
# # continue_train(model_name, seq_list, label_atrix, batch_size, epochs)
# # model_name= '../model/128/Model_5SrRNAUn80_Nopretrain.h5'
# start_predict(model_name, 0.9, 1.0, seq_list, label_atrix, batch_size)



