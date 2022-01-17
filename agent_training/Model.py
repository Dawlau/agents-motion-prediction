import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras

class Model:

	def __init__(self):
		self.model = models.Sequential()
		self.model.add(layers.Conv2D(1, (3, 3), activation = 'relu', input_shape = (11, 251 * 251, 3)))
		self.model.add(layers.Flatten())
		self.model.add(layers.Dense(2 * 80))
		self.optimizer = keras.optimizers.SGD(learning_rate = 1e-3)
		self.loss_fn = keras.losses.MeanSquaredError()
		self.epochs = 1

		print(self.model.summary())

	def train(self, batch):
		x_data, y_data = batch

		for epoch in range(self.epochs):
			for x, y in zip(x_data, y_data):
				with tf.GradientTape() as tape:
					logits = self.model(x.flatten().reshape(-1, 11, 251 * 251, 3), training = True)
					loss_value = self.loss_fn(y, logits)

				grads = tape.gradient(loss_value, self.model.trainable_weights)
				self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

				print(loss_value)
