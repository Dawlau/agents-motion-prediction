class Road:

	colors = {
		1 : (0, 0, 0),
		2 : (0.1, 0, 0),
		3 : (0.635, 0, 0),
		6 : (0.180, 0.180, 0.180),
		7 : (0.635, 0.635, 0.635),
		8 : (0.909, 0.909, 0.909),
		9 : (0.909, 0.909, 0),
		10 : (0.819, 0.819, 0),
		11 : (0.545, 0.545, 0),
		12 : (0.454, 0.454, 0),
		13 : (0.635, 0.635, 0),
		15 : (0.4, 0.2, 0),
		16 : (0.274, 0, 0.101),
		17 : (1, 0.180, 0.180),
		18 : (0.501, 0.501, 0.501),
		19 : (0.501, 0.501, 0)
	}

	def __init__(self, id, type, polyline):
		self.id = id
		self.type = type
		self.polyline = polyline
		self.color = Road.colors[self.type]

	def get_color(self):
		return self.color

	def get_id(self):
		return self.id

	def get_type(self):
		return self.type

	def get_polyline(self):
		return self.polyline