import tensorflow as tf
import numpy as np
import sys

class Rasterizer:

	image_width = 250
	image_height = 250

	@staticmethod
	def get_boundaries(roadgraph, traffic_lights, agents):
		x_min = sys.float_info.max
		y_min = sys.float_info.max
		x_max = -sys.float_info.max
		y_max = -sys.float_info.max

		roads = [road.get_polyline() for road in roadgraph]
		for road in roads:
			for vertex in road:
				x, y = vertex

				x_min = min(x_min, x)
				x_max = max(x_max, x)
				y_min = min(y_min, y)
				y_max = max(y_max, y)

		return x_min, y_min, x_max, y_max

	@staticmethod
	def get_normalized_coordinates(vertex, x_min, y_min, x_max, y_max):
		x, y = vertex
		real_width = x_max - x_min
		real_height = y_max - y_min

		x = (x - x_min - real_width / 2) * Rasterizer.image_width / real_width + Rasterizer.image_width / 2
		y = (y - y_min - real_height / 2) * Rasterizer.image_height / real_height + Rasterizer.image_height / 2

		return x, y

	@staticmethod
	def rasterize(roadgraph, traffic_lights, target_agent, surrounding_agents):
		pass
		# print(target_agent.shape)
		# x_min, y_min, x_max, y_max = Rasterizer.get_boundaries(roadgraph, traffic_lights, agents)

		# image = np.array(np.zeros(shape = (Rasterizer.image_width + 1, Rasterizer.image_height + 1, 3)))
		# for i in range(Rasterizer.image_width):
		# 	for j in range(Rasterizer.image_height):
		# 		image[i][j] = (1, 1, 1)

		# for road in roadgraph:
		# 	for vertex in road.get_polyline():
		# 		x, y = Rasterizer.get_normalized_coordinates(vertex, x_min, y_min, x_max, y_max)
		# 		x = int(x)
		# 		y = int(y)
		# 		image[x][y] = road.get_color()

		# for traffic_light in traffic_lights:
		# 	x, y = Rasterizer.get_normalized_coordinates((traffic_light.get_x(), traffic_light.get_y()), x_min, y_min, x_max, y_max)
		# 	x = int(x)
		# 	y = int(y)
		# 	for i in range(x - 1, x + 1):
		# 		for j in range(y - 1, y + 1):
		# 			image[i][j] = traffic_light.get_color()

		# for agent in agents:
		# 	vertices = agent.get_bbox()
		# 	for vertex in vertices:
		# 		x, y = Rasterizer.get_normalized_coordinates(vertex, x_min, y_min, x_max, y_max)

		# 		x = int(x)
		# 		y = int(y)

		# 		image[x][y] = agent.get_color()


		# import matplotlib.pyplot as plt
		# import time
		# plt.imsave(f'image{time.time()}.png', image)
		# return image
