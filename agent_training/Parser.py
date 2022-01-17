from TrafficLight import TrafficLight
from Agent import Agent
import numpy as np
import tensorflow as tf
from Road import Road

roadgraph_features = {
	'roadgraph_samples/id':
		tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
	'roadgraph_samples/type':
		tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
	'roadgraph_samples/valid':
		tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
	'roadgraph_samples/xyz':
		tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of agents.
state_features = {
	'state/id':
		tf.io.FixedLenFeature([128], tf.float32, default_value=None),
	'state/type':
		tf.io.FixedLenFeature([128], tf.float32, default_value=None),
	'state/is_sdc':
		tf.io.FixedLenFeature([128], tf.int64, default_value=None),
	'state/current/bbox_yaw':
		tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
	'state/current/length':
		tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
	'state/current/valid':
		tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
	'state/current/width':
		tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
	'state/current/x':
		tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
	'state/current/y':
		tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
	'state/future/bbox_yaw':
		tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
	'state/future/length':
		tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
	'state/future/valid':
		tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
	'state/future/width':
		tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
	'state/future/x':
		tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
	'state/future/y':
		tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
	'state/past/bbox_yaw':
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
	'state/past/length':
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
	'state/past/valid':
		tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
	'state/past/width':
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
	'state/past/x':
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
	'state/past/y':
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
	'traffic_light_state/current/state':
		tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
	'traffic_light_state/current/valid':
		tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
	'traffic_light_state/current/x':
		tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
	'traffic_light_state/current/y':
		tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
	'traffic_light_state/past/state':
		tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
	'traffic_light_state/past/valid':
		tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
	'traffic_light_state/past/x':
		tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
	'traffic_light_state/past/y':
		tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
	'traffic_light_state/future/state':
		tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
	'traffic_light_state/future/valid':
		tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
	'traffic_light_state/future/x':
		tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
	'traffic_light_state/future/y':
		tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
}

class Parser:

	def __init__(self, FILENAME):
		self.features_description = {}
		self.features_description.update(roadgraph_features)
		self.features_description.update(state_features)
		self.features_description.update(traffic_light_features)

		self.dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

	def generate_training_data(self):


		for data in self.dataset.as_numpy_iterator():
			scenario = tf.io.parse_single_example(data, self.features_description)

			roadgraph = Parser.parse_roadgraph(scenario)

			past_traffic_lights = Parser.parse_traffic_lights(scenario, [
				'traffic_light_state/past/valid',
				'traffic_light_state/past/state',
				'traffic_light_state/past/x',
				'traffic_light_state/past/y'
			])

			current_traffic_lights = Parser.parse_traffic_lights(scenario, [
				'traffic_light_state/current/valid',
				'traffic_light_state/current/state',
				'traffic_light_state/current/x',
				'traffic_light_state/current/y'
			])

			x_traffic_lights = past_traffic_lights + current_traffic_lights

			past_agents = Parser.parse_agents(scenario, [
				"state/type",
				"state/is_sdc",
				"state/past/valid",
				"state/past/bbox_yaw",
				"state/past/length",
				"state/past/width",
				"state/past/x",
				"state/past/y",
			])

			current_agents = Parser.parse_agents(scenario, [
				"state/type",
				"state/is_sdc",
				"state/current/valid",
				"state/current/bbox_yaw",
				"state/current/length",
				"state/current/width",
				"state/current/x",
				"state/current/y",
			])

			x_agents = past_agents + current_agents

			sdc_future_trajectory = Parser.get_sdc_future_trajectory(scenario)

			yield roadgraph, (x_agents, x_traffic_lights), sdc_future_trajectory
			# break

	@staticmethod
	def get_sdc_future_trajectory(scenario):
		is_sdc_mask = scenario["state/is_sdc"].numpy() > 0
		x = scenario["state/future/x"].numpy()[is_sdc_mask][0]
		y = scenario["state/future/y"].numpy()[is_sdc_mask][0]
		return np.concatenate((x, y))


	@staticmethod
	def parse_roadgraph(scenario):
		roadgraph = np.column_stack((
			scenario["roadgraph_samples/valid"].numpy(),
			scenario["roadgraph_samples/id"].numpy(),
			scenario["roadgraph_samples/type"].numpy(),
			scenario["roadgraph_samples/xyz"].numpy()
		))

		roadgraph = np.delete(roadgraph, roadgraph.shape[-1] - 1, axis = 1)
		roadgraph = roadgraph[roadgraph[ : , 0] > 0]
		roadgraph = np.delete(roadgraph, 0, axis = 1)

		roads = {} # {id, {type, [polyline]}}

		for _id, _type, _x, _y in roadgraph:
			_id = int(_id)
			_type = int(_type)
			if _id not in roads:
				roads[_id] = (_type, [(_x, _y)])
			else:
				roads[_id][1].append((_x, _y))

		for _id in roads.keys():
			roads[_id][1].append((roads[_id][1][0])) # close the polyline

		road = []
		for _id in roads.keys():
			road.append(Road(
				_id,
				roads[_id][0],
				roads[_id][1]
			))

		return road

	@staticmethod
	def parse_traffic_lights(scenario, features):

		valid_feature, state_feature, x_feature, y_feature = features

		data_valid = scenario[valid_feature].numpy()
		data_state = scenario[state_feature].numpy()
		data_x 	   = scenario[x_feature].numpy()
		data_y     = scenario[y_feature].numpy()

		traffic_lights = []
		for i in range(len(data_valid)):
			valid = data_valid[i]
			state = data_state[i]
			x 	  = data_x[i]
			y 	  = data_y[i]

			traffic_lights_frame = []
			for j in range(len(valid)):
				if valid[j]:
					traffic_lights_frame.append(TrafficLight(state[j], x[j], y[j]))

			traffic_lights.append(traffic_lights_frame)

		return traffic_lights

	@staticmethod
	def parse_agents(scenario, features):

		type_feature, is_sdc_feature, valid_feature, bbox_yaw_feature, length_feature, width_feature, x_feature, y_feature = features

		data_valid	  = scenario[valid_feature].numpy()
		data_type	  = scenario[type_feature].numpy()
		data_is_sdc   = scenario[is_sdc_feature].numpy()
		data_bbox_yaw = scenario[bbox_yaw_feature].numpy()
		data_length	  = scenario[length_feature].numpy()
		data_width 	  = scenario[width_feature].numpy()
		data_x		  = scenario[x_feature].numpy()
		data_y		  = scenario[y_feature].numpy()

		agents = [[] for _ in range(len(data_valid[0]))]
		for i in range(len(data_valid)):
			valid 	 = data_valid[i]
			type 	 = data_type[i]
			is_sdc   = data_is_sdc[i]
			bbox_yaw = data_bbox_yaw[i]
			length 	 = data_length[i]
			width 	 = data_width[i]
			x 		 = data_x[i]
			y 		 = data_y[i]

			for j in range(len(bbox_yaw)):
				if valid[j]:
					agents[j].append(Agent(type, bbox_yaw[j], length[j], width[j], x[j], y[j], is_sdc))

		return agents