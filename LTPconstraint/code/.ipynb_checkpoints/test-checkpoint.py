import tensorflow as tf
import numpy as np

# 列出所有可用設備
physical_devices = tf.config.list_physical_devices('GPU')
print("可用的 GPU 設備:", physical_devices)
strategy = tf.distribute.MirroredStrategy()
# 創建一個隨機的 tensor
random_tensor = tf.random.normal([1000, 1000])

# 在 GPU 上進行計算
result = tf.reduce_sum(random_tensor)

print("計算結果:", result.numpy())