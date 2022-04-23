import carla


class LaneMarking(object):

	def __init__(self, id_, type_, color, location, is_left):
		self.id = id_
		self.type = type_
		self.color = color
		self.location = location
		self.is_left = is_left


	def __hash__(self):
		return hash(hash(self.type) + hash(self.color) + hash(self.location) + hash(self.is_left))


	def __eq__(self, other):
		return self.id == other.id


	def get_id(self):
		return self.id


	def get_location(self):
		return self.location


	def get_type(self):
		if self.type is carla.LaneMarkingType.Solid:
			if self.color is carla.LaneMarkingColor.Yellow:
				return 11
			elif self.color is carla.LaneMarkingColor.White:
				return 7
		elif self.type is carla.LaneMarkingType.Broken:
			if self.color is carla.LaneMarkingColor.Yellow:
				return 9
			elif self.color is carla.LaneMarkingColor.White:
				return 6
		elif self.type is carla.LaneMarkingType.SolidSolid:
			if self.color is carla.LaneMarkingColor.Yellow:
				return 12
			elif self.color is carla.LaneMarkingColor.White:
				return 8
		return Exception("Invalid lane marking type or color")