# Simulation algorithms for human and car behavior


## Introduction

This repository summarizes the research I have done in the domain of autonomous agents, at the University of Bucharest.

More information about each implementation aspect can be found in the [official report](official_report.pdf) that I have compiled.

## Install requirements
Make sure you use python3.7 as the carla package for the Python API is not supported for later versions.
```bash
pip install -r requirements.txt
```
Before running the client, install the carla simulator. More information on their official documentation website: https://carla.readthedocs.io/en/latest/start_quickstart/. This implementation has been done for the CARLA simulator version 0.9.13. Although it should be backwards compatible for now, future updates to the simulator might come with breaking changes.

One more thing to mention is that you need to have the CARLA "agents" module in your path, which is not by default. You can either do that through a terminal session, or go into carla_client/config.py and change the variable CARLA_AGENTS_MODULE_PATH.

## Available maps by default in the CARLA simulator

```
Town01_Opt, Town01, Town02, Town02_Opt, Town03, Town03_Opt, Town04, Town04_Opt, Town05, Town05_Opt, Town10HD, Town10HD_Opt
```

## How to run

```bash
python carla_client/carla_client.py \
--backbone xception71 \
--spawn_index 12 \
--map Town05 \
--future_frames 20 \
--frames_to_skip 1
```

## Pausing the simulation

You can always pause the simulation by pressing 'p'. In contrast, pressing 'r' resumes it.
