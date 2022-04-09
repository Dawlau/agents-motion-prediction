import os
import sys
import carla
import random
import time
import pygame
from DataParser import DataParser
import math
from config import *


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


def main():

	pygame.init()

	try:

		client = carla.Client("localhost", 2000)
		client.set_timeout(2.0)

		# print(client.get_available_maps())
		# world = client.load_world("Town01_Opt")

		vehicles, walkers = init_scenario(client)
		ego_agent, ego_agent_camera = spawn_ego_agent(client)
		ego_agent.apply_control(carla.VehicleControl(throttle=1000, brake=0))
		ego_agent.set_autopilot(True)

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


		start_time = time.time()
		while True:
			# move spectator with ego agent
			world.get_spectator().set_transform(ego_agent_camera.get_transform())

			clock.tick()
			fps = clock.get_fps()


			current_time = time.time()
			if current_time - start_time >= SAMPLING_RATE:
				data_parser.update()
				print(data_parser.parsed["state/current/x"][0])
				print(f"Overhead: {current_time - start_time}")
				print(f"FPS: {fps}")
				start_time = current_time
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

