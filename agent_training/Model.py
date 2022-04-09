import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ReCoAt(tf.keras.Model):

	def __init__(self):
		super(ReCoAt, self).__init__()

		self.resnet50 = tf.keras.applications.resnet50.ResNet50(
			input_shape=(240, 240, 3),
			include_top=False)
		self.resnet50_output = tf.keras.Sequential([
			layers.Dense(128, activation="elu"),
			layers.Dropout(0.6)
		])

		self.target_agent_trajectory_encoder = tf.keras.Sequential([
			tf.keras.layers.Conv1D(64, 3),
			tf.keras.layers.LSTM(128)
		])

		self.surrounding_agents_trajectory_encoders = [tf.keras.Sequential([
			tf.keras.layers.Conv1D(64, 3),
			tf.keras.layers.LSTM(128)
		]) for i in range(10)]



	def call(self,
		environment_context: tf.Tensor,
		target_agent: tf.Tensor,
		surrounding_agents: tf.Tensor
		) -> tf.Tensor:

		environment_features = tf.keras.applications.resnet.preprocess_input(environment_context)
		environment_features = self.resnet50(environment_features)
		environment_features = self.resnet50_output(environment_features)

		target_agent_features = self.target_agent_trajectory_encoder(target_agent)

		# encoded_trajectories = tf.convert_to_tensor([])
		# encoded_trajectories = tf.expand_dims(encoded_trajectories, axis=0)
		# print(encoded_trajectories.shape)
		for i in range(surrounding_agents.shape[1]):
			agent = surrounding_agents[ : , i, ...]
			agent = tf.squeeze(agent, axis=0)
			print(agent.shape)
			# print(self.surrounding_agents_trajectory_encoders[i](agent).shape)
			# encoded_trajectories = tf.concat(
				# [encoded_trajectories, self.surrounding_agents_trajectory_encoders[i](agent)], 0)

		return target_agent_features
