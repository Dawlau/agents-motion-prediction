import numpy as np
import carla
from LaneMarking import LaneMarking
from config import *
import math
import time


class DataParser:

	def __init__(self, world, world_map, ego_agent, agents):
		self.world 			= world
		self.world_map 		= world_map
		self.ego_agent 		= ego_agent
		self.agents 		= agents
		self.agents			= self.agents + [None] * (NUM_MAX_AGENTS - 1 - len(agents))
		self.roadgraph 		= self.world_map.generate_waypoints(distance=DISTANCE_BETWEEN_WAYPOINTS)
		self.parsed = {}
		self.init_parsed_dict()


	def init_key(self, key, shape):
		self.parsed[key] = np.empty(shape)


	def init_parsed_dict(self):
		self.parsed["scenario/id"] = "0"
		self.init_traffic_lights_data()
		self.init_agents_data()
		self.init_roadgraph_data()


	def append_to_key(self, key, value):
		self.parsed[key] = np.append(self.parsed[key], value, axis=0)


	def init_traffic_lights_data(self):

		for key, shape in zip(traffic_lights_keys, traffic_lights_keys_shapes):
			self.init_key(key, shape)

		for traffic_light in self.get_traffic_lights():
			valid = np.array([[traffic_light is not None]])
			if traffic_light is None:
				id_   = np.array([[-1]])
				state = np.array([[-1]])
			else:
				road_id, traffic_light_state = traffic_light
				id_   = np.array([[road_id]])
				state = np.array([[traffic_light_state]])

			values = [id_, state, valid]

			for key, value in zip(traffic_lights_keys, values):
				self.append_to_key(key, value)

		for key in traffic_lights_keys:
			self.parsed[key] = self.parsed[key].T


	def init_agents_data(self):

		for key, shape in zip(agents_keys, agents_keys_shapes):
			self.init_key(key, shape)

		for i, agent in enumerate([self.ego_agent] + self.agents):
			if agent is not None:
				bbox_yaw = math.pi * agent.get_transform().rotation.yaw / 180
				past_x 			 = np.array([[agent.get_transform().location.x] * NUM_PAST_FRAMES])
				past_y 			 = np.array([[agent.get_transform().location.y] * NUM_PAST_FRAMES])
				past_speed 		 = np.array([[0] * NUM_PAST_FRAMES])
				past_vel_yaw 	 = np.array([[bbox_yaw] * NUM_PAST_FRAMES])
				past_bbox_yaw 	 = np.array([[bbox_yaw] * NUM_PAST_FRAMES])
				current_x 		 = np.array([[agent.get_transform().location.x]])
				current_y 		 = np.array([[agent.get_transform().location.y]])
				current_speed 	 = np.array([[0]])
				current_vel_yaw  = np.array([[bbox_yaw]])
				current_bbox_yaw = np.array([[bbox_yaw]])
				current_width 	 = np.array([[2 * agent.bounding_box.extent.x]])
				current_length 	 = np.array([[2 * agent.bounding_box.extent.y]])
				future_x   		 = np.array([[-1] * NUM_FUTURE_FRAMES])
				future_y 		 = np.array([[-1] * NUM_FUTURE_FRAMES])
				past_valid 		 = np.array([[1] * NUM_PAST_FRAMES])
				current_valid 	 = np.array([[1]])
				future_valid 	 = np.array([[-1] * NUM_FUTURE_FRAMES])
				agent_id 		 = np.array([[ agent.id ]])
				agent_type 		 = np.array([[ 1 if isinstance(agent, carla.libcarla.Vehicle) else 2 ]])
				track_to_predict = np.array([[ 1 if i == 0 else 0 ]])
			else:
				past_x 			 = np.array([[-1] * NUM_PAST_FRAMES])
				past_y 			 = np.array([[-1] * NUM_PAST_FRAMES])
				past_speed 		 = np.array([[-1] * NUM_PAST_FRAMES])
				past_vel_yaw 	 = np.array([[-1] * NUM_PAST_FRAMES])
				past_bbox_yaw 	 = np.array([[-1] * NUM_PAST_FRAMES])
				current_x 		 = np.array([[-1]])
				current_y 		 = np.array([[-1]])
				current_speed 	 = np.array([[-1]])
				current_vel_yaw  = np.array([[-1]])
				current_bbox_yaw = np.array([[-1]])
				current_width 	 = np.array([[-1]])
				current_length 	 = np.array([[-1]])
				future_x   		 = np.array([[-1] * NUM_FUTURE_FRAMES])
				future_y 		 = np.array([[-1] * NUM_FUTURE_FRAMES])
				past_valid 		 = np.array([[0] * NUM_PAST_FRAMES])
				current_valid 	 = np.array([[0]])
				future_valid 	 = np.array([[0] * NUM_FUTURE_FRAMES])
				agent_id 		 = np.array([[-1]])
				agent_type 		 = np.array([[-1]])
				track_to_predict = np.array([[-1]])

			init_values = [
				past_x, past_y, past_speed, past_vel_yaw, past_bbox_yaw,
				current_x, current_y, current_speed, current_vel_yaw,
				current_bbox_yaw, current_width, current_length,
				future_x, future_y,
				past_valid, current_valid, future_valid,
				agent_id, agent_type, track_to_predict,
			]

			for key, value in zip(agents_keys, init_values):
				self.append_to_key(key, value)

		self.parsed["state/id"] 			   = np.squeeze(self.parsed["state/id"])
		self.parsed["state/type"] 			   = np.squeeze(self.parsed["state/type"])
		self.parsed["state/tracks_to_predict"] = np.squeeze(self.parsed["state/tracks_to_predict"])


	def init_roadgraph_data(self):

		for key, shape in zip(roadgraph_keys, roadgraph_keys_shapes):
			self.init_key(key, shape)

		# road line waypoints

		for waypoint in self.roadgraph:
			if waypoint.lane_type is carla.LaneType.Driving:
				location = waypoint.transform.location

				id_   = np.array([[int(waypoint.road_id)]])
				type_ = np.array([[1]])
				valid = np.array([[1]])
				xyz   = np.array([[location.x, location.y, location.z]])

				values = [id_, type_, valid, xyz]

				for key, value in zip(roadgraph_keys, values):
					self.append_to_key(key, value)


		# stop signs

		stop_signs = self.get_landmarks_by_id("206")
		stop_sign_base_id = np.max(self.parsed["roadgraph_samples/id"])

		if len(stop_signs):
			for stop_sign in stop_signs:
				location = stop_sign.get_location()

				id_   	 = np.array([[int(stop_sign_base_id + stop_sign.id)]])
				type_ 	 = np.array([[17]])
				valid 	 = np.array([[1]])
				xyz 	 = np.array([[location.x, location.y, location.z]])

				values = [id_, type_, valid, xyz]

				for key, value in zip(roadgraph_keys, values):
					self.append_to_key(key, value)


		# crosswalks

		crosswalk_id = np.max(self.parsed["roadgraph_samples/id"])
		crosswalk_locations = self.world_map.get_crosswalks()

		if len(crosswalk_locations):
			start_location = crosswalk_locations[0]
			for i, location in enumerate(crosswalk_locations[0 : ]):
				if i + 1 >= len(crosswalk_locations):
					id_   	 = np.array([[int(crosswalk_id)]])
					type_ 	 = np.array([[18]])
					valid 	 = np.array([[1]])
					xyz 	 = np.array([[location.x, location.y, location.z]])

					values = [id_, type_, valid, xyz]

					for key, value in zip(roadgraph_keys, values):
						self.append_to_key(key, value)

					break

				if location == start_location:
					crosswalk_id += 1
					start_location = crosswalk_locations[i + 1] if i else crosswalk_locations[0]

				id_   	 = np.array([[int(crosswalk_id)]])
				type_ 	 = np.array([[18]])
				valid 	 = np.array([[1]])
				xyz 	 = np.array([[location.x, location.y, location.z]])

				values = [id_, type_, valid, xyz]

				for key, value in zip(roadgraph_keys, values):
					self.append_to_key(key, value)


		# lane markings

		base_id = np.max(self.parsed["roadgraph_samples/id"]) + 1
		lane_markings = self.get_lane_markings(base_id)

		for lane_marking in lane_markings:
			location = lane_marking.get_location()

			id_   = np.array([[int(lane_marking.get_id())]])
			type_ = np.array([[lane_marking.get_type()]])
			valid = np.array([[1]])
			xyz   = np.array([[location.x, location.y, location.z]])

			values = [id_, type_, valid, xyz]

			for key, value in zip(roadgraph_keys, values):
				self.append_to_key(key, value)

		# padding

		for key in roadgraph_keys:
			if self.parsed[key].shape[0] >= MAX_ROADGRAPH_SAMPLES_NUM:
				continue

			if key == "roadgraph_samples/xyz":
				padding_value = [[-1, -1, -1]]
			elif key == "roadgraph_samples/valid":
				padding_value = [[0]]
			else:
				padding_value = [[-1]]

			self.parsed[key] = np.append(
				self.parsed[key],
				padding_value * max(MAX_ROADGRAPH_SAMPLES_NUM - len(self.parsed[key]), 0),
				axis=0
			)

		self.parsed["roadgraph_samples/type"] = self.parsed["roadgraph_samples/type"].astype(np.int32)


	def get_landmarks_by_id(self, id_):

		landmarks = self.world_map.get_all_landmarks_of_type(id_)

		landmarks = {
			landmark.id: landmark for landmark in landmarks
		}.values()

		true_landmarks = []
		for lmrk in landmarks:
			if id_ == "206":
				landmark = self.world.get_traffic_sign(lmrk)
			else:
				landmark = self.world.get_traffic_light(lmrk)
			if landmark:
				true_landmarks.append(landmark)

		return list(true_landmarks)


	def get_traffic_lights(self):

		def distance_comparison(traffic_light):
			traffic_light_x = traffic_light.get_location().x
			traffic_light_y = traffic_light.get_location().y
			ego_agent_x 	= self.ego_agent.get_location().x
			ego_agent_y 	= self.ego_agent.get_location().y

			return (ego_agent_x - traffic_light_x) ** 2 + (ego_agent_y - traffic_light_y) ** 2


		def distance_filter(traffic_light):
			traffic_light_x = traffic_light.get_location().x
			traffic_light_y = traffic_light.get_location().y
			ego_agent_x 	= self.ego_agent.get_location().x
			ego_agent_y 	= self.ego_agent.get_location().y

			distance = math.sqrt((ego_agent_x - traffic_light_x) ** 2 + (ego_agent_y - traffic_light_y) ** 2)

			return distance <= MAX_TRAFFIC_LIGHT_DISTANCE

		traffic_lights = self.get_landmarks_by_id("1000001")

		traffic_lights = list(filter(distance_filter, traffic_lights))
		traffic_lights.sort(key=distance_comparison)

		affected_roads = []

		for traffic_light in traffic_lights:
			for affected_road in traffic_light.get_affected_lane_waypoints():
				affected_roads.append((affected_road.road_id, hash(traffic_light.state) + 4))

		affected_roads = affected_roads[ : MAX_TRAFFIC_LIGHTS_NUM]
		affected_roads = affected_roads + [None] * (MAX_TRAFFIC_LIGHTS_NUM - len(traffic_lights))

		return affected_roads


	def get_lane_markings(self, base_id):
		right_lane_markings = []
		left_lane_markings = []

		for waypoint in self.roadgraph:
			lane_location = waypoint.transform.location

			right_vector = waypoint.transform.get_right_vector()

			right_marking_location = lane_location + waypoint.lane_width / 2 * right_vector
			left_marking_location = lane_location - waypoint.lane_width / 2 * right_vector

			if waypoint.right_lane_marking.type is not carla.LaneMarkingType.NONE:
				marking_id = waypoint.road_id + 2 * base_id

				right_lane_markings.append(LaneMarking(
					marking_id,
					waypoint.right_lane_marking.type,
					waypoint.right_lane_marking.color,
					right_marking_location,
					0
				))

			if waypoint.left_lane_marking.type is not carla.LaneMarkingType.NONE:
				marking_id = waypoint.road_id + base_id

				left_lane_markings.append(LaneMarking(
					marking_id,
					waypoint.left_lane_marking.type,
					waypoint.left_lane_marking.color,
					left_marking_location,
					1
				))

		lane_markings = left_lane_markings + right_lane_markings
		lane_markings.sort(key=lambda x: x.get_id())

		return lane_markings


	def update_agents_info(self):
		for i, agent in enumerate([self.ego_agent] + self.agents):
			if agent is None:
				break

			location  = agent.get_transform().location

			current_x = np.array([location.x])
			current_y = np.array([location.y])

			location 	  = np.array([location.x, location.y])
			prev_location = np.array([self.parsed["state/past/x"][i][-1], self.parsed["state/past/y"][i][-1]])

			current_speed 	 = np.array([np.linalg.norm(location - prev_location) / SAMPLING_RATE])
			bbox_yaw = math.pi * agent.get_transform().rotation.yaw / 180

			if current_speed < MIN_SPEED:
				vel_yaw = bbox_yaw
			else:
				velocity = agent.get_velocity()
				vel_yaw = math.atan2(velocity.y, velocity.x)

			current_vel_yaw  = np.array([vel_yaw])
			current_bbox_yaw = np.array([bbox_yaw])

			values = [current_x, current_y, current_speed, current_vel_yaw, current_bbox_yaw]

			for past_key, current_key, value in zip(agents_keys[ : TO_UPDATE_INFO // 2], agents_keys[TO_UPDATE_INFO // 2 : TO_UPDATE_INFO], values):
				self.parsed[past_key][i] = np.append(
					self.parsed[past_key][i, 1 : ],
					self.parsed[current_key][i],
					axis=0
				)
				self.parsed[current_key][i] = value


	def update(self):
		self.update_agents_info()
		self.init_traffic_lights_data()