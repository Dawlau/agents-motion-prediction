class Frame:

	def __init__(self, road, traffic_lights, agents):
		self.road = road
		self.traffic_lights = traffic_lights
		self.agents = agents

	def get_road(self):
		return self.road

	def get_traffic_lights(self):
		return self.traffic_lights

	def get_agents(self):
		return self.agents