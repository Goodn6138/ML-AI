import tensorflow as tf

# Define a function
def f(x):
    return tf.sin(x) + x**2

# Define a value of x
x = tf.constant(2.0)

# Create a gradient tape context
with tf.GradientTape(persistent=True) as tape:
    # Record operations on x
    tape.watch(x)
    y = f(x)

# Compute the first derivative with respect to x
dy_dx = tape.gradient(y, x)

# Compute the second derivative with respect to x
d2y_dx2 = tape.gradient(dy_dx, x)

# Clean up the persistent gradient tape
del tape

print("Value of y:", y.numpy())
print("Value of dy/dx:", dy_dx.numpy())
print("Value of d2y/dx2:", d2y_dx2.numpy())
