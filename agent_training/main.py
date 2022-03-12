from Rasterizer import Rasterizer
from Parser import Parser
import os

from DataLoader import DataLoader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
# from Model import Model

TRAINING_DATA_PATH = "/home/dawlau/waymo_dataset/training"

def main() -> None:
	DataLoader(TRAINING_DATA_PATH).generate_data()

if __name__ == "__main__":
	main()
	print(time.process_time())
# training_data_files = [os.path.join(TRAINING_DATA_PATH, file) for file in os.listdir(TRAINING_DATA_PATH)]
# print("Num GPUs Available: ", device_lib.list_local_devices())

# model = Model()

# for file in training_data_files[ : 1]:
# 	parser = Parser(file)

# 	x_data = []
# 	y_data = []

# 	for roadgraph, (x_agents, x_traffic_lights), sdc_future_trajectory in parser.generate_training_data():
# 		x = np.array([Rasterizer.rasterize_frame(roadgraph, x_traffic_lights[i], x_agents[i]) for i in range(len(x_agents))])
# 		y = sdc_future_trajectory
# 		x_data.append(x)
# 		y_data.append(y)
# 		break

	# x_data = np.array(x_data)
	# y_data = np.array(y_data)

	# model.train((x_data, y_data))

"""
ideas:
	maybe make number of surrounding agents a hyperparameter
Data parsing:
- What I need:
	for every file:
		for every scenario in file:
			for every index in tracks_to_predict:
				train model for the current agent
				input:
					environment context - tensor of size (IM_WIDTH, IM_HEIGHT, 3 - rgb channels)
					ego agent features - tensor of size (batch_size, features_no = 5)
					surrounding agents features tensor of size (agents_no, batch_size, features_no = 5)
				output:
					hyperparameter(maybe?) K trajectories along with scores

Environment context processing:
input:
	ego agent trajectory: tensor of size (batch_size, 5 - x y vx vy bbox_yaw)
	surrounding agent trajectories (subsampled to use only the relevant surrounding agents aka closest to ego agent): tensor of size (agents_no, batch_size, 5)
	traffic_lights: tensor of size (traffic_lights_no)
	roadgraph: tensor of size (map_features_no, 3 - x y type) ??
output:
	rgb image
notes:
	rotate and translate the scene such that the ego agent is always in the lower left corner (almost) as specified in the report to allow model consistency in learning and prediction
	don't forget the tail for the agents
	maybe in the end add potential trajectories as well wtf are those
"""