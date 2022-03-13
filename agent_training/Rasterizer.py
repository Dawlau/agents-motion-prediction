import tensorflow as tf
import numpy as np
import sys
from matplotlib import pyplot as plt


class Rasterizer:

	IMAGE_WIDTH = 240
	IMAGE_HEIGHT = 240
	TARGET_AGENT_COLOR = (255, 0, 0) # red
	BACKGROUND_COLOR = (0, 0, 0) # black
	TARGET_AGENT_CENTER = (IMAGE_WIDTH / 5,	IMAGE_HEIGHT / 2)
	SURROUNDING_AGENTS_COLORS = [
		(255, 0, 255), # magenta
		(0, 0, 255), # blue
		(0, 255, 0) # green
	]

	@staticmethod
	def show_image(environment_context: np.array) -> None:
		environment_context = environment_context.transpose((1, 0, 2))
		plt.imshow(environment_context, origin="lower")
		plt.show()

	@staticmethod
	def rasterize(
		roadgraph: tf.Tensor,
		traffic_lights: tf.Tensor,
		target_agent: tf.Tensor,
		surrounding_agents: tf.Tensor,
		surrounding_agents_types: tf.Tensor) -> tf.Tensor:

		"""
			document the function
			add function annotations
			Notes:
				- delete invalid coords for agents
				- filter for points in boundaries
				- rotations
				- bboxes
				- circles as traffic lights
				- how the fuck do I do streets?
				- continuous to discrete
		"""

		environment_context = np.full(
			shape=(Rasterizer.IMAGE_WIDTH, Rasterizer.IMAGE_HEIGHT, 3),
			fill_value=Rasterizer.BACKGROUND_COLOR)

		past_surrounding_agents    = surrounding_agents[ : , : -1, : ]
		current_surrounding_agents = surrounding_agents[-1]

		past_target_agent 	 = target_agent[ : -1]
		current_target_agent = target_agent[-1]

		translation = tf.math.subtract(
			Rasterizer.TARGET_AGENT_CENTER, current_target_agent[ : -1])


		# draw tails

		past_surrounding_agents_xy = past_surrounding_agents[..., : -1]
		past_target_agent_xy 	   = past_target_agent[..., : -1]

		past_surrounding_agents_xy += translation
		past_target_agent_xy 	   += translation

		past_surrounding_agents_xy = tf.cast(
			past_surrounding_agents_xy, dtype=tf.int32)
		past_target_agent_xy	   = tf.cast(
			past_target_agent_xy, dtype=tf.int32)

		environment_context[tuple(tf.transpose(past_target_agent_xy))] = Rasterizer.TARGET_AGENT_COLOR

		for i, x in enumerate(past_surrounding_agents_xy):
			agent_type = surrounding_agents_types[i] - 1
			agent_type = int(agent_type)
			environment_context[tuple(tf.transpose(x))] = Rasterizer.SURROUNDING_AGENTS_COLORS[agent_type]

		Rasterizer.show_image(environment_context)

		# print(target_agent_theta)
		# print(roadgraph.shape)
		# print(traffic_lights.shape)
		# print(target_agent.shape)
		# print(surrounding_agents.shape)
		# print(surrounding_agents_types.shape)
		# print("------------------------------------------")

		# print(target_agent.shape)
		# x_min, y_min, x_mplt.axis(), y_max = Rasterizer.get_boundaries(roadgraph, traffic_lights, agents)

		# image = np.array(np.zeros(shape = (Rasterizer.image_width + 1, Rasterizer.image_height + 1, 3)))
		# for i in range(Rasterizer.image_width):
		# 	for j in range(Rasterizer.image_height):
		# 		image[i][j] = (1, 1, 1)

		# for road in roadgraph:
		# 	for vertex in road.get_polyline():
		# 		x, y = Rasterizer.get_normalized_coordinates(vertex, x_min, y_min, x_mplt.axis(), y_max)
		# 		x = int(x)
		# 		y = int(y)
		# 		image[x][y] = road.get_color()

		# for traffic_light in traffic_lights:
		# 	x, y = Rasterizer.get_normalized_coordinates((traffic_light.get_x(), traffic_light.get_y()), x_min, y_min, x_mplt.axis(), y_max)
		# 	x = int(x)
		# 	y = int(y)
		# 	for i in range(x - 1, x + 1):
		# 		for j in range(y - 1, y + 1):
		# 			image[i][j] = traffic_light.get_color()

		# for agent in agents:
		# 	vertices = agent.get_bbox()
		# 	for vertex in vertices:
		# 		x, y = Rasterizer.get_normalized_coordinates(vertex, x_min, y_min, x_mplt.axis(), y_max)

		# 		x = int(x)
		# 		y = int(y)

		# 		image[x][y] = agent.get_color()


		# import matplotlib.pyplot as plt
		# import time
		# plt.imsave(f'image{time.time()}.png', image)
		# return image
