from Rasterizer import Rasterizer
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