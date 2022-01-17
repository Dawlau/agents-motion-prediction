from Rasterizer import Rasterizer
from Parser import Parser
import os
import time
import numpy as np
from Model import Model

TRAINING_DATA_PATH = "/tmp/agents-motion-prediction/waymo_dataset/training/"

training_data_files = [os.path.join(TRAINING_DATA_PATH, file) for file in os.listdir(TRAINING_DATA_PATH)]

model = Model()

for file in training_data_files[ : 1]:
	parser = Parser(file)

	x_data = []
	y_data = []

	for roadgraph, (x_agents, x_traffic_lights), sdc_future_trajectory in parser.generate_training_data():
		x = np.array([Rasterizer.rasterize_frame(roadgraph, x_traffic_lights[i], x_agents[i]) for i in range(len(x_agents))])
		y = sdc_future_trajectory
		x_data.append(x)
		y_data.append(y)
		break

	x_data = np.array(x_data)
	y_data = np.array(y_data)

	model.train((x_data, y_data))