import os
import sys
import carla
import random
import time
import pygame
from DataParser import DataParser
import math
from config import *
from RasterOnCNN import prerender
import torch
import numpy as np
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
			walker_controller.start()
			walker_controller.go_to_location(world.get_random_location_from_navigation())
			walkers.append(walker)

	return vehicles, walkers


def spawn_ego_agent(client):

	world = client.get_world()
	blueprint_library = world.get_blueprint_library()
	bp = random.choice(blueprint_library.filter('vehicle'))

	ego_agent = None

	while ego_agent is None:
		transform = random.choice(world.get_map().get_spawn_points())
		ego_agent = world.try_spawn_actor(bp, transform)

	camera_bp = world.get_blueprint_library().find('sensor.other.collision')
	camera_transform = carla.Transform(carla.Location(x=-7, z=3))
	ego_agent_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_agent)

	return ego_agent, ego_agent_camera


def get_trajectory(model, parsed, ego_agent):
	data   = prerender.merge(parsed, False)[0]

	raster = data["raster"].astype(np.float32)
	raster = raster.transpose(2, 1, 0) / 255
	raster = torch.tensor(raster)
	raster = torch.unsqueeze(raster, dim=0).cuda()

	confidences_logits, logits = model(raster)
	confidences_logits = torch.squeeze(confidences_logits)
	logits = torch.squeeze(logits)
	trajectory = logits[torch.argmax(confidences_logits, dim=0)].cpu().detach().numpy()

	yaw = data["yaw"]

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


def main():

	pygame.init()

	try:

		client = carla.Client("localhost", 2000)
		client.set_timeout(2.0)

		# print(client.get_available_maps())
		# world = client.load_world("Town02")

		vehicles, walkers = init_scenario(client)
		ego_agent, ego_agent_camera = spawn_ego_agent(client)
		# ego_agent.set_autopilot(True)

		world = client.get_world()
		world_map = world.get_map()

		clock = pygame.time.Clock()

		settings = world.get_settings()
		settings.synchronous_mode = True
		settings.fixed_delta_seconds = 1 / 200
		# settings.no_rendering_mode = True
		settings.no_rendering_mode = False
		world.apply_settings(settings)


		data_parser = DataParser(
			world=world,
			world_map=world_map,
			ego_agent=ego_agent,
			agents=vehicles + walkers
		)

		model = torch.jit.load(MODEL_PATH).cuda().eval()

		waypoints = []
		current_waypoint_idx = 0

		control = VehiclePIDController(
			ego_agent,
			args_lateral = {'K_P': 1, 'K_D': 0.8, 'K_I': 0.8, 'dt': 1.0 / 10.0},
			args_longitudinal = {'K_P': 1, 'K_D': 0.8, 'K_I': 0.8, 'dt': 1.0 / 10.0}
		)
		speed = 15

		start_time = time.time()
		while True:
			# move spectator with ego agent
			world.get_spectator().set_transform(ego_agent_camera.get_transform())

			clock.tick()

			for waypoint in waypoints:
				location = waypoint.transform.location
				draw_location = carla.Location(
					location.x,
					location.y,
					location.z + 2,
				)
				world.debug.draw_string(draw_location, "o", draw_shadow=False,
											 color=carla.Color(r=255, g=0, b=0), life_time=0.01)

			if time.time() - start_time >= SAMPLING_RATE:
				data_parser.update()

			if current_waypoint_idx == len(waypoints):
				trajectory = get_trajectory(model, data_parser.parsed, ego_agent)
				waypoints = []

				for vertex in trajectory:
					waypoint = carla.Location(
						vertex[0],
						vertex[1],
						ego_agent.get_location().z
					)
					waypoints.append(world_map.get_waypoint(waypoint))


				current_waypoint_idx = 0

			ego_agent.apply_control(control.run_step(speed, waypoints[current_waypoint_idx]))

			# check dist
			if euclidean_distance(ego_agent.get_location(), waypoints[current_waypoint_idx].transform.location) < DISTANCE_THRESHOLD:
				current_waypoint_idx += 1

			fps = clock.get_fps()
			print(f"Overhead: {time.time() - start_time}")
			print(f"FPS: {fps}")
			start_time = time.time()

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

