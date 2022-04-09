
import tensorflow as tf

roadgraph_features = {
	'roadgraph_samples/type':
		tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
	'roadgraph_samples/xyz':
		tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
	"roadgraph_samples/id":
		tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
	"roadgraph_samples/valid":
		tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None)
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
	"state/id":
		tf.io.FixedLenFeature([128], tf.float32, default_value=None),
	"scenario/id": tf.io.FixedLenFeature([1], tf.string, default_value=None)
}

traffic_light_features = {
	'traffic_light_state/current/state':
		tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
	'traffic_light_state/current/x':
		tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
	'traffic_light_state/current/y':
		tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
	"traffic_light_state/current/id": tf.io.FixedLenFeature(
        [1, 16], tf.int64, default_value=None
    ),
}


features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)

def get_dist(a, b):
	x1, y1 = a
	x2, y2 = b

	return tf.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


scenarios = tf.data.TFRecordDataset("/home/dawlau/waymo_dataset/training/training_tfexample.tfrecord-00000-of-01000", compression_type='')
for scenario in scenarios.as_numpy_iterator():
	data = tf.io.parse_single_example(scenario, features_description)
	print(data["roadgraph_samples/xyz"].shape)
	# x = data["state/current/x"]
	# y = data["state/current/y"]

	# vertices = list(zip(x, y))
	# dist = [get_dist(vertices[0], vertices[i]) for i in range(1, len(vertices)) if data["state/current/valid"][i]]

	# print(max(dist))
	# for i, vertex in zip(x, y):
		# print(get_dist(vertex))
		# break

	# break


