import os
import sys
import numpy as np
import torch
import random

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

NUM_MAX_AGENTS = 128
NUM_PAST_FRAMES = 10
NUM_FUTURE_FRAMES = 80
DISTANCE_BETWEEN_WAYPOINTS = 0.5
MAX_TRAFFIC_LIGHTS_NUM = 16
MAX_ROADGRAPH_SAMPLES_NUM = 20000
SAMPLING_RATE = 0.1
DISTANCE_THRESHOLD = 0.5
MIN_SPEED = 0.2
MAX_TRAFFIC_LIGHT_DISTANCE = 50.0

VEHICLES_NO = 2
WALKERS_NO  = 0

TO_UPDATE_INFO = 10

RASTER_CNN_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
# MODEL_PATH = os.path.join(RASTER_CNN_PATH, "RasterOnCNN_Resnet18", "resnet18.pt")
MODEL_PATH = os.path.join(RASTER_CNN_PATH, "RasterOnCNN_Xception71", "test_xception71_263000_dev_131.pth")

CARLA_AGENTS_MODULE_PATH = "/opt/carla-simulator/PythonAPI/carla"

agents_keys = [
	"state/past/x",
	"state/past/y",
	"state/past/speed",
	"state/past/vel_yaw",
	"state/past/bbox_yaw",
	"state/current/x",
	"state/current/y",
	"state/current/speed",
	"state/current/vel_yaw",
	"state/current/bbox_yaw",
	"state/current/width",
	"state/current/length",
	"state/future/x",
	"state/future/y",
	"state/past/valid",
	"state/current/valid",
	"state/future/valid",
	"state/id",
	"state/type",
	"state/tracks_to_predict",
]


agents_keys_shapes = [
	(0, NUM_PAST_FRAMES),
	(0, NUM_PAST_FRAMES),
	(0, NUM_PAST_FRAMES),
	(0, NUM_PAST_FRAMES),
	(0, NUM_PAST_FRAMES),
	(0, 1),
	(0, 1),
	(0, 1),
	(0, 1),
	(0, 1),
	(0, 1),
	(0, 1),
	(0, NUM_FUTURE_FRAMES),
	(0, NUM_FUTURE_FRAMES),
	(0, NUM_PAST_FRAMES),
	(0, 1),
	(0, NUM_FUTURE_FRAMES),
	(0, 1),
	(0, 1),
	(0, 1),
]


roadgraph_keys = [
	"roadgraph_samples/id",
	"roadgraph_samples/type",
	"roadgraph_samples/valid",
	"roadgraph_samples/xyz",
]


roadgraph_keys_shapes = [
	(0, 1),
	(0, 1),
	(0, 1),
	(0, 3),
]


traffic_lights_keys = [
	"traffic_light_state/current/id",
	"traffic_light_state/current/state",
	"traffic_light_state/current/valid",
]


traffic_lights_keys_shapes = [
	(0, 1),
	(0, 1),
	(0, 1),
]

sys.path.append(RASTER_CNN_PATH)
sys.path.append(CARLA_AGENTS_MODULE_PATH)