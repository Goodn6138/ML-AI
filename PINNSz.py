import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


k = 0.003
model = tf.keras.Sequential([tf.keras.layers.Dense(10 , activation = "relu" , input_shape = (2,)),
                             tf.keras.layers.Dense(10 , activation = "relu"),
                             tf.keras.layers.Dense(1)])

model.compile()

def derivatives():
        
    
