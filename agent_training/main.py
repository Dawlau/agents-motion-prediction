from Rasterizer import Rasterizer
import os

from DataLoader import DataLoader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from Model import ReCoAt


TRAINING_DATA_PATH = "/home/dawlau/waymo_dataset/training"

def main() -> None:
	# DataLoader(TRAINING_DATA_PATH).generate_data()
	dataset = tf.data.Dataset.from_generator(
		DataLoader(TRAINING_DATA_PATH).generate_data,
		output_signature=(
			tf.TensorSpec(shape=(240, 240, 3), dtype=tf.int8),
			tf.TensorSpec(shape=(10, 5), dtype=tf.float32),
			tf.TensorSpec(shape=(10, 10, 5), dtype=tf.float32)
		))
	dataset = dataset.batch(1)

	model = ReCoAt()
	for batch in dataset:
		environment_context, target_agent, surrounding_agents = batch

		print(model(environment_context, target_agent, surrounding_agents))
		break

if __name__ == "__main__":
	main()
	print(time.process_time())