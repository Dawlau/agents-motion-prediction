import tensorflow as tf
import numpy as np
import sys
from matplotlib import pyplot as plt
import cv2
import math


class Rasterizer:

	IMAGE_WIDTH = 240
	IMAGE_HEIGHT = 240
	TARGET_AGENT_COLOR = (255, 0, 0) # red
	BACKGROUND_COLOR = (0, 0, 0) # black
	TARGET_AGENT_CENTER = (IMAGE_WIDTH / 5,	IMAGE_HEIGHT / 2)
	SURROUNDING_AGENTS_COLORS = [
		(0, 0, 0),
		(255, 0, 255), # magenta
		(0, 0, 255), # blue
		(0, 255, 0), # green
		(0, 0, 0)
	]
	AGENTS_SIZES = [0, 10, 4, 5, 0]
	TRAFFIC_LIGHTS_COLORS = [
		(0, 0, 0), 		# Unknown
		(255, 0, 0),	# Arrow_Stop
		(255, 0, 0),	# Arrow_Caution
		(0, 255, 0),	# Arrow_Go
		(255, 0, 0), 	# Stop
		(255, 0, 0), 	# Caution
		(0, 255, 0), 	# Go
		(255, 0, 0), 	# Flashing_Stop
		(255, 0, 0)		# Flashing_Caution
	]
	TRAFFIC_LIGHTS_RADIUS = 5

	origin_translation = None
	global_rotation = None
	image_translation = None


	@staticmethod
	def show_image(environment_context: np.array) -> None:
		plt.imshow(environment_context, origin="lower")
		plt.show()


	@staticmethod
	def rotate_bbox(
		bbox: tf.Tensor,
		bbox_center: tf.Tensor,
		angle: tf.Tensor) -> tf.Tensor:

		rotation_matrix = tf.convert_to_tensor([
			[math.cos(angle), -math.sin(angle)],
			[math.sin(angle), math.cos(angle)]
		])

		result = tf.matmul(
					rotation_matrix,
					tf.transpose(bbox - bbox_center))
		result = tf.transpose(result) + bbox_center
		return result


	@staticmethod
	def get_bbox_offset(agent_type: tf.Tensor) -> tf.Tensor:
		size = Rasterizer.AGENTS_SIZES[agent_type]
		bbox_offset =	tf.convert_to_tensor([
							[-1, -2],
							[size, -2],
							[size, 2],
							[-1, 2]
						], dtype=tf.float32)
		return bbox_offset


	@staticmethod
	def draw_target_agent(
		environment_context: np.array,
		past_target_agent: tf.Tensor,
		current_target_agent: tf.Tensor,
		target_agent_type: tf.Tensor) -> np.array:

		"""
			draw target agent information into the environment context
			inputs:
				environment_context: tensor with shape (240, 240, 3)
				past_target_agent: tensor with shape (VALID_PAST_FRAMES_NO, 3)
				current_target_agent: tensor with shape (1, 3)
			output:
				updated environment_context
		"""

		# draw tails
		past_coords = past_target_agent[ : , : -1]
		norm_past_coords = Rasterizer.apply_global_transformation(past_coords)
		norm_past_coords = tf.math.round(norm_past_coords)
		norm_past_coords = tf.cast(norm_past_coords, dtype=tf.int32).numpy()
		environment_context = cv2.polylines(
			environment_context, [norm_past_coords], False, Rasterizer.TARGET_AGENT_COLOR, lineType=cv2.LINE_AA)

		# draw bbox
		current_coords = current_target_agent[ : , : -1]
		heading_angle = current_target_agent[ : , -1]
		size = Rasterizer.AGENTS_SIZES[target_agent_type]
		bbox = tf.math.add(
			current_coords,
			Rasterizer.get_bbox_offset(target_agent_type)
		)
		bbox = Rasterizer.rotate_bbox(bbox, current_coords, heading_angle)
		bbox = Rasterizer.apply_global_transformation(bbox)
		bbox = tf.math.round(bbox)
		bbox = tf.cast(bbox, dtype=tf.int32).numpy()

		environment_context = cv2.fillPoly(
			environment_context, [bbox], Rasterizer.TARGET_AGENT_COLOR, lineType=cv2.LINE_AA)

		return environment_context


	@staticmethod
	def draw_surrounding_agents(
		environment_context: np.array,
		past_surrounding_agents: tf.Tensor,
		current_surrounding_agents: tf.Tensor,
		surrounding_agents_types: tf.Tensor) -> np.array:

		for i, agent_type in enumerate(surrounding_agents_types):
			past_states = past_surrounding_agents[i]
			current_state = current_surrounding_agents[i]

			past_states_mask = past_states != [-1., -1., -1.]
			past_states_mask = tf.reduce_any(past_states_mask, axis=1)
			past_states = past_states[past_states_mask]

			past_coords = past_states[ : , : -1]
			norm_past_coords = Rasterizer.apply_global_transformation(past_coords)
			norm_past_coords = tf.math.round(norm_past_coords)
			norm_past_coords = tf.cast(norm_past_coords, dtype=tf.int32).numpy()
			environment_context = cv2.polylines(
				environment_context, [norm_past_coords], False, Rasterizer.SURROUNDING_AGENTS_COLORS[tf.cast(agent_type, dtype=tf.int32)], lineType=cv2.LINE_AA)

			# draw bbox
			current_coords = current_state[ : -1]
			heading_angle = current_state[-1]
			size = Rasterizer.AGENTS_SIZES[tf.cast(agent_type, dtype=tf.int32)]
			bbox = tf.math.add(
				current_coords,
				Rasterizer.get_bbox_offset(tf.cast(agent_type, dtype=tf.int32))
			)
			bbox = Rasterizer.rotate_bbox(bbox, current_coords, heading_angle)
			bbox = Rasterizer.apply_global_transformation(bbox)
			bbox = tf.math.round(bbox)
			bbox = tf.cast(bbox, dtype=tf.int32).numpy()

			environment_context = cv2.fillPoly(
				environment_context, [bbox], Rasterizer.SURROUNDING_AGENTS_COLORS[tf.cast(agent_type, dtype=tf.int32)], lineType=cv2.LINE_AA)

		return environment_context


	@staticmethod
	def apply_global_transformation(coords: tf.Tensor) -> tf.Tensor:
		"""
			input:
				coords: tensor with shape (x, 2)
			output:
				coords after applying global transformations
		"""
		angle = Rasterizer.global_rotation
		rotation_matrix = tf.convert_to_tensor([
			[math.cos(angle), -math.sin(angle)],
			[math.sin(angle), math.cos(angle)]
		])

		result = coords + Rasterizer.origin_translation
		result = tf.linalg.matmul(
			rotation_matrix, tf.transpose(result))
		result = tf.transpose(result) - Rasterizer.origin_translation
		result = result + Rasterizer.image_translation

		return result


	@staticmethod
	def draw_traffic_lights(
		environment_context: np.array,
		traffic_lights: tf.Tensor) -> np.array:

		for traffic_light in traffic_lights:
			x, y, state = tf.unstack(traffic_light)
			state = tf.cast(state, dtype=tf.int32)
			xy = tf.concat([x, y], axis=0)
			xy = Rasterizer.apply_global_transformation(xy)
			xy = tf.math.round(xy)
			xy = tf.cast(xy, dtype=tf.int32)
			xy = tf.squeeze(xy, axis=0).numpy()

			environment_context = cv2.circle(
				environment_context, xy, Rasterizer.TRAFFIC_LIGHTS_RADIUS, Rasterizer.TRAFFIC_LIGHTS_COLORS[state], thickness=-1, lineType=cv2.LINE_AA)

		return environment_context


	@staticmethod
	def draw_road(
		environment_context: np.array,
		road_xy: tf.Tensor,
		road_type: tf.Tensor) -> np.array:

		_type = road_type.numpy()
		xy = Rasterizer.apply_global_transformation(road_xy)
		xy = tf.math.round(xy)
		xy = tf.cast(xy, dtype=tf.int32).numpy()

		if _type <= 3:
			environment_context = cv2.polylines(
				environment_context, [xy], isClosed=False, color=(128, 128, 128), thickness=2, lineType=cv2.LINE_AA)
		elif _type == 6:
			pass
		elif 7 <= _type <= 8:
			environment_context = cv2.polylines(
				environment_context, [xy], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
		elif 9 <= _type <= 13:
			environment_context = cv2.polylines(
				environment_context, [xy], isClosed=False, color=(255, 255, 128), thickness=1, lineType=cv2.LINE_AA)
		elif 15 <= _type <= 16:
			environment_context = cv2.polylines(
				environment_context, [xy], isClosed=False, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
		elif _type == 17:
			print(xy)
			environment_context = cv2.circle(
				environment_context, xy, Rasterizer.TRAFFIC_LIGHTS_RADIUS, Rasterizer.TRAFFIC_LIGHTS_COLORS[state], thickness=-1, lineType=cv2.LINE_AA)
		elif _type == 18:
			environment_context = cv2.fillPoly(
				environment_context, [xy], color=(0, 0, 255), lineType=cv2.LINE_AA)
		else:
			environment_context = cv2.fillPoly(
				environment_context, [xy], color=(255, 165, 0), lineType=cv2.LINE_AA)

		return environment_context

	@staticmethod
	def draw_roadgraph(
		environment_context: np.array,
		roadgraph: tf.Tensor) -> np.array:

		current_start = 0
		current_type  = roadgraph[0][-1]

		for i, road in enumerate(roadgraph):
			if road[-1] != current_type:
				# last_env = environment_context.copy()
				current_road = roadgraph[current_start : i]
				current_road_xy = current_road[..., : -1]
				current_road_type = current_road[-1, -1]
				environment_context = Rasterizer.draw_road(
					environment_context, current_road_xy, current_road_type)
				current_start = i
				current_type  = road[-1]

				# if current_road_type.numpy() <= 3 and not np.array_equal(last_env, environment_context):
				# 	return environment_context

		current_road = roadgraph[current_start : -1]
		current_road_xy = current_road[..., : -1]
		current_road_type = current_road[-1, -1]
		environment_context = Rasterizer.draw_road(
			environment_context, current_road_xy, current_road_type)


		return environment_context

	@staticmethod
	def rasterize(
		roadgraph: tf.Tensor,
		traffic_lights: tf.Tensor,
		target_agent: tf.Tensor,
		target_agent_type: tf.Tensor,
		surrounding_agents: tf.Tensor,
		surrounding_agents_types: tf.Tensor) -> tf.Tensor:

		"""
			document the function
		"""

		environment_context = np.full(
			shape=(Rasterizer.IMAGE_WIDTH, Rasterizer.IMAGE_HEIGHT, 3),
			fill_value=Rasterizer.BACKGROUND_COLOR)
		environment_context = environment_context.astype(np.uint8)

		past_surrounding_agents    = surrounding_agents[ : , : -1, : ]
		current_surrounding_agents = surrounding_agents[ : , -1]

		past_target_agent 	 = target_agent[ : -1]
		current_target_agent = tf.expand_dims(target_agent[-1], axis=0)

		# clean target coords
		past_target_agent_mask = tf.reduce_any(
			past_target_agent != [-1., -1., -1.], axis=1)
		past_target_agent = past_target_agent[past_target_agent_mask]

		Rasterizer.image_translation = tf.math.subtract(
			Rasterizer.TARGET_AGENT_CENTER, current_target_agent[ : , : -1])
		Rasterizer.origin_translation = -current_target_agent[ : , : -1]

		Rasterizer.global_rotation = -current_target_agent[ : , -1]

		environment_context = Rasterizer.draw_roadgraph(environment_context, roadgraph)

		environment_context = Rasterizer.draw_target_agent(
			environment_context, past_target_agent, current_target_agent, target_agent_type)

		environment_context = Rasterizer.draw_surrounding_agents(
			environment_context, past_surrounding_agents, current_surrounding_agents, surrounding_agents_types)

		environment_context = Rasterizer.draw_traffic_lights(environment_context, traffic_lights)

		
		return environment_context