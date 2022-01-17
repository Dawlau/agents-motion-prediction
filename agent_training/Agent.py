import math
import numpy as np

class Agent:

	sdc_color = (1, 0.725, 0.796)

	colors = {
		1 : (0.305, 0.956, 0.811),
		2 : (0.682, 0.305, 0.956),
		3 : (0.074, 0.082, 0.847)
	}

	def __init__(self, type, bbox_yaw, length, width, x, y, is_sdc):
		self.type = type
		self.bbox_yaw = math.pi / 2 - bbox_yaw
		self.length = length
		self.width = width
		self.x = x
		self.y = y
		self.is_sdc = is_sdc
		self.color = Agent.sdc_color if self.is_sdc else Agent.colors[self.type]

	def get_color(self):
		return self.color

	def get_bbox(self):
		rotation_matrix = np.array([
			[math.cos(self.bbox_yaw), -math.sin(self.bbox_yaw)],
			[math.sin(self.bbox_yaw), math.cos(self.bbox_yaw)]
		])

		def float_to_int(x):
			sign = np.sign(x)
			positive_x = abs(x)

			if positive_x - int(positive_x) > 0.5:
				return int(x) + int(sign)
			else:
				return int(x)

		up_left = np.array([
			-self.width / 2,
			self.length / 2
		])
		lower_right = np.array([
			self.width / 2,
			-self.length / 2
		])

		vertices = []
		for x in range(float_to_int(up_left[0]), float_to_int(lower_right[0]) + 1):
			for y in range(float_to_int(lower_right[1]), float_to_int(up_left[1]) + 1):
				vertex = np.array([x, y])
				vertex = rotation_matrix @ vertex
				vertex = np.array([
					vertex[0] + self.x,
					vertex[1] + self.y
				])
				vertices.append(vertex)

		return vertices

	def get_type(self):
		return self.type

	def get_bbox_yaw(self):
		return self.bbox_yaw

	def get_length(self):
		return self.length

	def get_width(self):
		return self.width

	def get_x(self):
		return self.x

	def get_y(self):
		return self.y