class TrafficLight:

	colors = {
		0 : (0, 0, 0),
		1 : (0.8, 0, 0),
		2 : (1, 0.901, 0),
		3 : (0, 0.901, 0.035),
		4 : (1, 0, 0),
		5 : (1, 1, 0),
		6 : (0, 1, 0.035),
		7 : (0.8, 0, 0),
		8 : (1, 0.8, 0)
	}

	def __init__(self, state, x, y):
		self.state = state
		self.x = x
		self.y = y
		self.color = TrafficLight.colors[state]

	def get_color(self):
		return self.color

	def get_x(self):
		return self.x

	def get_state(self):
		return self.state

	def get_y(self):
		return self.y

	def get_color(self):
		return (0.5, 0.2, 0.5)