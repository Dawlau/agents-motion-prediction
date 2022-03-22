import os
import numpy as np
import tensorflow as tf
from typing import Generator, Dict, Iterable, Tuple
from Rasterizer import Rasterizer

roadgraph_features = {
	'roadgraph_samples/type':
		tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
	'roadgraph_samples/xyz':
		tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of agents.
state_features = {
	'state/type':
		tf.io.FixedLenFeature([128], tf.float32, default_value=None),
	'state/current/bbox_yaw':
		tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
	'state/current/valid':
		tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
	'state/current/x':
		tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
	'state/current/y':
		tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
	'state/future/x':
		tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
	'state/future/y':
		tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
	"state/past/velocity_x":
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
	"state/past/velocity_y":
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
	'state/past/bbox_yaw':
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
	'state/past/valid':
		tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
	'state/past/x':
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
	'state/past/y':
		tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
	'state/tracks_to_predict':
		tf.io.FixedLenFeature([128], tf.int64, default_value=None),
}

traffic_light_features = {
	'traffic_light_state/current/state':
		tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
	'traffic_light_state/current/x':
		tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
	'traffic_light_state/current/y':
		tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
}

DISTANCE_THRESHOLD = 30.0
SURROUNDING_AGENTS_NO = 10

class DataLoader:


	@classmethod
	def __init__(self, data_path: str) -> None:
		self.data_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
		self.features_description = {}
		self.features_description.update(roadgraph_features)
		self.features_description.update(state_features)
		self.features_description.update(traffic_light_features)

		self.agent_past_features = [
			"state/past/x",
			"state/past/y",
			"state/past/bbox_yaw",
			"state/past/velocity_x",
			"state/past/velocity_y",
		]

		self.agent_current_features = [
			"state/current/x",
			"state/current/y",
			"state/current/bbox_yaw",
		]

		self.agent_future_features = [
			"state/future/x",
			"state/future/y"
		]


	@staticmethod
	def get_agents_features(data: Dict, features: Iterable[str]) -> tf.Tensor:
		agent_features = [data[feature] for feature in features]
		agent_features = tf.stack(agent_features, -1)
		return agent_features


	@classmethod
	def parse_scenario(self,
		data: Dict) -> Tuple[
			tf.Tensor, tf.Tensor, tf.Tensor,
			tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

		past_is_valid = data["state/past/valid"] > 0
		current_is_valid = data["state/current/valid"] > 0

		valid_agents_mask = tf.reduce_any(
			tf.concat([past_is_valid, current_is_valid], 1), 1)

		agents_to_predict = data["state/tracks_to_predict"] > 0
		agent_types = data["state/type"][valid_agents_mask]

		past_agents = DataLoader.get_agents_features(data, self.agent_past_features)[valid_agents_mask]
		current_agents = DataLoader.get_agents_features(data, self.agent_current_features)[valid_agents_mask]
		future_agents = DataLoader.get_agents_features(data, self.agent_future_features)[valid_agents_mask]

		roadgraph_coords = data["roadgraph_samples/xyz"]
		roadgraph_type = tf.cast(data["roadgraph_samples/type"], dtype=tf.float32)
		roadgraph = tf.concat(
			[roadgraph_coords, roadgraph_type], -1)
		roadgraph_mask = roadgraph_type > 0
		roadgraph = roadgraph[tf.squeeze(roadgraph_mask)]

		traffic_lights_x = tf.transpose(data["traffic_light_state/current/x"])
		traffic_lights_y = tf.transpose(data["traffic_light_state/current/y"])
		traffic_lights_state = tf.transpose(data["traffic_light_state/current/state"])
		traffic_lights_mask = traffic_lights_state > 0
		traffic_lights_state = tf.cast(traffic_lights_state, dtype=tf.float32)
		traffic_lights = tf.concat(
			[traffic_lights_x, traffic_lights_y, traffic_lights_state], 1)
		traffic_lights = traffic_lights[tf.squeeze(traffic_lights_mask)]

		return past_agents, current_agents, agents_to_predict, agent_types, roadgraph, traffic_lights, future_agents


	@classmethod
	def get_surrounding_agents(self,
		past_surrounding_agents: tf.Tensor,
		current_surrounding_agents: tf.Tensor,
		current_target_agent: tf.Tensor,
		surrounding_agents_types: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

		# prepare features for distance filtering
		# remove heading angle
		target_agent_xy 	  = current_target_agent[..., : -1]
		surrounding_agents_xy = current_surrounding_agents[..., : -1]

		# reshape such that new tensor is of shape (agents_no, 2)
		target_agent_xy = tf.reshape(
			target_agent_xy, shape=(target_agent_xy.shape[0], target_agent_xy.shape[-1]))

		surrounding_agents_xy = tf.reshape(
			surrounding_agents_xy, shape=(surrounding_agents_xy.shape[0], surrounding_agents_xy.shape[-1]))

		# filter agents that are farther from the target agent than the threshold
		distances = tf.norm(
			tf.math.subtract(surrounding_agents_xy, target_agent_xy), axis=-1)

		# remove agents that are farther than needed
		distances_mask 			   = distances <= DISTANCE_THRESHOLD
		past_surrounding_agents    = past_surrounding_agents[distances_mask]
		current_surrounding_agents = current_surrounding_agents[distances_mask]
		surrounding_agents_xy 	   = surrounding_agents_xy[distances_mask]

		# calculate distances of relevant agents and get the closest ones
		distances = tf.norm(
			tf.math.subtract(surrounding_agents_xy, target_agent_xy), axis=-1)

		best_agents = tf.math.top_k(
			-distances, k=min(SURROUNDING_AGENTS_NO, distances.shape[-1])).indices

		# these are going to be the relevant surrounding agents
		past_surrounding_agents = tf.gather(
			past_surrounding_agents, best_agents)
		current_surrounding_agents = tf.gather(
			current_surrounding_agents, best_agents)

		# remove the farthest ones
		surrounding_agents_types = surrounding_agents_types[distances_mask]
		# get the closest ones
		surrounding_agents_types = tf.gather(
			surrounding_agents_types, best_agents)

		return past_surrounding_agents, current_surrounding_agents, surrounding_agents_types


	@classmethod
	def generate_data(self) -> Generator[None, None, None]:

		for file in self.data_files[ : 1]:
			scenarios = tf.data.TFRecordDataset(file, compression_type='')

			for scenario in scenarios.as_numpy_iterator():
				data = tf.io.parse_single_example(scenario, self.features_description)

				past_agents, current_agents, agents_to_predict, agent_types, roadgraph, traffic_lights, future_agents = self.parse_scenario(data)

				for x in tf.where(agents_to_predict):
					idx = tf.squeeze(x)

					# features of target agent
					past_target_agent 	 = past_agents[idx]
					current_target_agent = current_agents[idx]
					target_agent_type 	 = tf.cast(agent_types[idx], dtype=tf.int32)

					# features of surrounding agents
					past_surrounding_agents    = tf.concat(
						[past_agents[ : idx], past_agents[idx + 1 : ]], 0)
					current_surrounding_agents = tf.concat(
						[current_agents[ : idx], current_agents[idx + 1 : ]], 0)
					surrounding_agents_types   = tf.concat(
						[agent_types[ : idx], agent_types[idx + 1 : ]], 0)

					past_surrounding_agents, current_surrounding_agents, surrounding_agents_types = self.get_surrounding_agents(past_surrounding_agents, current_surrounding_agents, current_target_agent, surrounding_agents_types)

					target_agent = past_target_agent
					# padding surrounding agents with zeros
					padding_size = SURROUNDING_AGENTS_NO - past_surrounding_agents.shape[0]
					surrounding_agents = tf.concat([
						past_surrounding_agents,
						tf.zeros((padding_size,) + past_surrounding_agents.shape[1 : ])], 0)

					# prepare data for environment context rasterizer
					# remove velocities and concat with current timestamp for surrounding agents
					env_context_surrounding_agents = tf.concat(
						[past_surrounding_agents[..., : 3], current_surrounding_agents], 1)

					env_context_target_agent = tf.concat(
						[past_target_agent[..., : 3], current_target_agent], 0)

					environment_context = Rasterizer.rasterize(
						roadgraph,
						traffic_lights,
						env_context_target_agent,
						target_agent_type,
						env_context_surrounding_agents,
						surrounding_agents_types
					)

					# yield environment_context, target_agent, surrounding_agents
					break


				break
