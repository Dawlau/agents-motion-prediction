from config import *
import os
import carla
import random
import time
import pygame
from DataParser import DataParser
import math
import torch
import numpy as np
import argparse
import timm
from agents.navigation.controller import VehiclePIDController


def init_scenario(client):

	vehicles = []
	walkers  = []

	world = client.get_world()
	blueprint_library = world.get_blueprint_library()

	for _ in range(VEHICLES_NO):
		bp = random.choice(blueprint_library.filter('vehicle'))

		transform = random.choice(world.get_map().get_spawn_points())
		vehicle = world.try_spawn_actor(bp, transform)

		while vehicle is None:
			vehicle = world.try_spawn_actor(bp, transform)

		if vehicle is not None:
			vehicles.append(vehicle)
			vehicle.set_autopilot(True)

	for _ in range(WALKERS_NO):
		bp = random.choice(blueprint_library.filter('walker.pedestrian*'))

		transformation = carla.Transform()
		transformation.location = world.get_random_location_from_navigation()
		transform.location.z += 1
		walker = world.try_spawn_actor(bp, transformation)
		world.wait_for_tick()

		if walker is not None:
			controller_bp = world.get_blueprint_library().find("controller.ai.walker")

			walker_controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
			world.wait_for_tick()

			while walker_controller is None:
				walker_controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
				world.wait_for_tick()

			walker_controller.start()
			walker_controller.go_to_location(world.get_random_location_from_navigation())
			walkers.append(walker)

	return vehicles, walkers


def spawn_ego_agent(client, spawn_index):

	world = client.get_world()
	blueprint_library = world.get_blueprint_library()
	bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))

	ego_agent = None

	world_map = world.get_map()

	while ego_agent is None:
		if spawn_index is None:
			index = random.choice(list(range(len(world_map.get_spawn_points()))))
		else:
			index = spawn_index
		transform = world_map.get_spawn_points()[index]
		ego_agent = world.try_spawn_actor(bp, transform)

	camera_bp = world.get_blueprint_library().find('sensor.other.collision')
	camera_transform = carla.Transform(carla.Location(x=-7, z=3))
	ego_agent_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_agent)

	return ego_agent, ego_agent_camera


def get_trajectory(model, parsed, ego_agent, merge, backbone):
	data   = merge(parsed)[0]

	raster = data["raster"].astype(np.float32)
	raster = raster.transpose(2, 1, 0) / 255
	raster = torch.tensor(raster)
	raster = torch.unsqueeze(raster, dim=0).cuda()

	if backbone == "resnet18":
		confidences_logits, logits = model(raster)
	else:
		outputs = model(raster)
		confidences_logits, logits = outputs[:,:N_TRAJS], outputs[:,N_TRAJS:]
		logits = logits.view(raster.shape[0], N_TRAJS, TL, 2)

	confidences_logits = torch.squeeze(confidences_logits)
	logits = torch.squeeze(logits)
	confidences_logits = torch.softmax(confidences_logits, dim=-1)
	trajectory = logits[torch.argmax(confidences_logits, dim=0)].cpu().detach().numpy()

	yaw = -data["yaw"]

	rot_matrix = np.array([
		[np.cos(yaw), -np.sin(yaw)],
		[np.sin(yaw), np.cos(yaw)],
	])

	x = ego_agent.get_location().x
	y = ego_agent.get_location().y

	trajectory = trajectory @ rot_matrix + np.array([x, y])

	return trajectory


def euclidean_distance(a, b):
	return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def parse_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--backbone",
		type=str,
		required=False,
		default="resnet18"
	)

	parser.add_argument(
		"--spawn_index",
		type=int,
		required=False,
		default=None
	)

	parser.add_argument(
		"--map",
		type=str,
		required=False,
		default=None
	)

	parser.add_argument(
		"--future_frames",
		type=int,
		required=False,
		default=NUM_FUTURE_FRAMES
	)

	parser.add_argument(
		"--frames_to_skip",
		type=int,
		required=False,
		default=1
	)

	args = parser.parse_args()

	if args.future_frames < 1 or args.future_frames > NUM_FUTURE_FRAMES:
		raise Exception("The number of future frames must be within the threshold")

	if args.frames_to_skip < 1 or args.frames_to_skip > NUM_FUTURE_FRAMES:
		raise Exception("The number of frames to skip must be valid")

	return args


def load_model(backbone):

	model_path = os.path.join(MODELS_PATH, f"{backbone}.pt")

	if backbone == "resnet18":
		model = torch.jit.load(model_path).cuda().eval()
		from prerender_resnet18 import merge
	elif backbone == "xception71":
		model = timm.create_model(
			"xception71",
			pretrained=True,
			in_chans=IN_CHANNELS,
			num_classes=N_TRAJS * 2 * TL + N_TRAJS,
		).cuda()
		model.load_state_dict(torch.load(model_path)["model_state_dict"])
		model.eval()
		from prerender_xception71 import merge
	else:
		raise Exception("Invalid backbone")

	return model, merge


def show_trajectory(world, waypoints, color):
	for waypoint in waypoints:
		draw_location = carla.Location(
			waypoint.transform.location.x,
			waypoint.transform.location.y,
			2
		)
		world.debug.draw_string(draw_location, "o", color=color, draw_shadow=False, life_time=0.01)


def main():

	args = parse_arguments()
	backbone = args.backbone
	spawn_index = args.spawn_index
	simulation_map = args.map
	future_frames = args.future_frames
	frames_to_skip = args.frames_to_skip

	pygame.init()

	try:

		client = carla.Client("localhost", 2000)
		client.set_timeout(2.0)

		if simulation_map is not None:
			world = client.load_world(simulation_map)

		vehicles, walkers = init_scenario(client)
		ego_agent, ego_agent_camera = spawn_ego_agent(client, spawn_index)

		world = client.get_world()
		world_map = world.get_map()

		clock = pygame.time.Clock()

		settings = world.get_settings()
		settings.synchronous_mode = True
		settings.fixed_delta_seconds = 1 / 200
		settings.no_rendering_mode = False
		world.apply_settings(settings)

		data_parser = DataParser(
			world=world,
			world_map=world_map,
			ego_agent=ego_agent,
			agents=vehicles + walkers
		)

		model, merge = load_model(backbone)

		waypoints = []
		current_waypoint_idx = 0

		control = VehiclePIDController(
			ego_agent,
			args_lateral = {'K_P': 1.0, 'K_D': 0.8, 'K_I': 0.8, 'dt': 1.0 / 10.0},
			args_longitudinal = {'K_P': 1.0, 'K_D': 0.8, 'K_I': 0.8, 'dt': 1.0 / 10.0}
		)
		speed = 0
		world.get_spectator().set_transform(ego_agent_camera.get_transform())
		start_time = time.time()
		while True:
			# move spectator with ego agent
			# world.get_spectator().set_transform(ego_agent_camera.get_transform())

			clock.tick()

			if time.time() - start_time >= SAMPLING_RATE:
				data_parser.update()
				fps = clock.get_fps()
				print(f"Overhead: {time.time() - start_time}")
				print(f"FPS: {fps}")
				start_time = time.time()

			# if the nextpoint has been reached
			if current_waypoint_idx < len(waypoints) and euclidean_distance(ego_agent.get_location(), waypoints[current_waypoint_idx].transform.location) < DISTANCE_THRESHOLD:
				current_waypoint_idx += 1

			# get new trajectory if all points have been reached
			if current_waypoint_idx >= len(waypoints):
				trajectory = get_trajectory(model, data_parser.parsed, ego_agent, merge, backbone)
				waypoints = []

				for vertex in trajectory[ : future_frames : frames_to_skip]:
					waypoint = carla.Location(
						vertex[0],
						vertex[1],
						ego_agent.get_location().z
					)
					waypoints.append(world_map.get_waypoint(waypoint))

				current_waypoint_idx = 0

			ego_agent.apply_control(control.run_step(speed, waypoints[current_waypoint_idx])) # move agent to the next waypoint

			destination = waypoints[-1].transform.location - ego_agent.get_location()
			bbox_yaw = data_parser.parsed["state/current/bbox_yaw"][0]
			if math.cos(bbox_yaw) * destination.x + math.sin(bbox_yaw) * destination.y < 0: # if the predicted trajectory is backwards
				show_trajectory(world, waypoints, carla.Color(r=0, g=255, b=0))
				waypoints = []
				current_waypoint_idx = 0
				speed = 0
			else:
				speed = 15
				show_trajectory(world, waypoints, carla.Color(r=255, g=0, b=0))


			world.tick()

	finally:
		client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
		client.apply_batch([carla.command.DestroyActor(x) for x in walkers])
		ego_agent_camera.destroy()
		client.apply_batch([carla.command.DestroyActor(ego_agent)])

		settings = world.get_settings()
		settings.synchronous_mode = False
		world.apply_settings(settings)


if __name__ == '__main__':
	main()