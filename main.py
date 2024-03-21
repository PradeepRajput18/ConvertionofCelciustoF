import tensorflow as tf
import numpy as np 

# Define input and output data
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Definining neural network architecture layers
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

# Compiling the model using Adams Optimizer
#mentioned to check indetail about adam algorithm
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
    loss = 'mean_squared_error'
)

# Training the model
results = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)

# Plot the training loss
import matplotlib.pyplot as plt

#graphical interface printing
plt.xlabel("# epochs")
plt.ylabel("Loss")
plt.plot(results.history["loss"])

res = model.predict([100.0])

print("The predicted temperature in Fahrenheit for 100 Celsius is " + str(res))
