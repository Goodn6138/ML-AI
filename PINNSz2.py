import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,), activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])

x = np.arange(0, 1, 0.002)
y = 2*np.exp(2.5*x**2) - 4/5

def loss_re(dy_dx, x, y):
    return tf.square(dy_dx - x * (5 * y + 4))

def loss_bc(y_pred):
    lower_bc = tf.square(y_pred[0] - 1.23)  # Lower boundary condition at x = 0
    upper_bc = tf.square(y_pred[-1] - 23.22)  # Upper boundary condition at x = 1
    return lower_bc + upper_bc

def cost(model, x):
    X = tf.constant(x.reshape(-1, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X)
        y = model(X)
    dy_dx = tape.gradient(y, X)

    loss_re_value = loss_re(dy_dx, X, y)
    y_pred = model(X)
    loss_bc_value = loss_bc(y_pred)

    return loss_re_value + loss_bc_value

model.compile(loss=cost)
model.fit(x, epochs=10, verbose=1)
