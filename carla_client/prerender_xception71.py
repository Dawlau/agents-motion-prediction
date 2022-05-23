"""
    This implementation has been adapted from the official repo of MotionCNN from the Waymo Open Dataset for Motion Prediction competition 2021
    Official github repo: https://github.com/kbrodt/waymo-motion-prediction-2021/tree/main
"""

import cv2
import numpy as np


road_colors = [
    "#122329",
    "#142C2B",
    "#153027",
    "#163320",
    "#173719",
    "#203B18",
    "#2D3E19",
    "#3C421A",
    "#463F1B",
    "#49351C",
    "#4D291D",
    "#511D20",
    "#551E31",
    "#591F44",
    "#5D2059",
    "#512061",
    "#402165",
    "#2D2269",
    "#222D6D",
    "#234571",
    "#236075",
]


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


road_colors = [hex_to_rgb(x) for x in road_colors]
road_colors = [int(x) for x in np.linspace(1, 255, len(road_colors)).astype("uint8")]


def rasterize(
    tracks_to_predict,
    past_x,
    past_y,
    current_x,
    current_y,
    current_yaw,
    past_yaw,
    past_valid,
    current_valid,
    agent_type,
    roadlines_coords,
    roadlines_types,
    roadlines_valid,
    roadlines_ids,
    widths,
    lengths,
    agents_ids,
    tl_states,
    tl_ids,
    tl_valids,
    future_x,
    future_y,
    future_valid,
    scenario_id,
    past_speed,
    current_speed,
    validate,
):

    GRES = []

    raster_size = 224
    shift = 2 ** 9
    displacement = np.array([[raster_size // 4, raster_size // 2]]) * shift

    tl_dict = {"green": set(), "yellow": set(), "red": set()}

    # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
    # Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
    for tl_state, tl_id, tl_valid in zip(
        tl_states.flatten(), tl_ids.flatten(), tl_valids.flatten()
    ):
        if tl_valid == 0:
            continue
        if tl_state in [1, 4, 7]:
            tl_dict["red"].add(tl_id)
        if tl_state in [2, 5, 8]:
            tl_dict["yellow"].add(tl_id)
        if tl_state in [3, 6]:
            tl_dict["green"].add(tl_id)

    XY = np.concatenate(
        (
            np.expand_dims(np.concatenate((past_x, current_x), axis=1), axis=-1),
            np.expand_dims(np.concatenate((past_y, current_y), axis=1), axis=-1),
        ),
        axis=-1,
    )  # * shift * 3

    GT_XY = np.concatenate(
        (np.expand_dims(future_x, axis=-1), np.expand_dims(future_y, axis=-1)), axis=-1
    )  # * shift * 3

    YAWS = np.concatenate((past_yaw, current_yaw), axis=1)

    agents_valid = np.concatenate((past_valid, current_valid), axis=1)

    speeds = np.concatenate([past_speed, current_speed], axis=1)

    speeds = np.clip(speeds / 30 * 255, 0, 255).astype(np.int)

    roadlines_valid = roadlines_valid.reshape(-1)
    roadlines_coords = (
        roadlines_coords[:, :2][roadlines_valid > 0] * shift * 3 * raster_size / 512
    )
    roadlines_types = roadlines_types[roadlines_valid > 0]
    roadlines_ids = roadlines_ids.reshape(-1)[roadlines_valid > 0]

    for i, (
        xy,
        current_val,
        val,
        a_type,
        yaw,
        agent_id,
        gt_xy,
        future_val,
        predict,
        _speeds,
    ) in enumerate(
        zip(
            XY,
            current_valid,
            agents_valid,
            agent_type,
            current_yaw.flatten(),
            agents_ids,
            GT_XY,
            future_valid,
            tracks_to_predict.flatten(),
            speeds,
        )
    ):
        if (not validate and future_val.sum() == 0) or (validate and predict == 0):
            continue
        if current_val == 0:
            continue
        RES_ROADMAP = np.ones((raster_size, raster_size, 3), dtype=np.uint8) * 255
        RES_EGO = [
            np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
            for _ in range(11 * 2)
        ]
        RES_OTHER = [
            np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
            for _ in range(11 * 2)
        ]

        xy_val = xy[val > 0]
        if len(xy_val) == 0:
            continue

        unscaled_center_xy = xy_val[-1].reshape(1, -1)
        center_xy = unscaled_center_xy * shift * 3 * raster_size / 512
        rot_matrix = np.zeros((2, 2), dtype=np.float32)

        rot_matrix = np.array(
            ((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw)))
        ).reshape(2, 2)

        centered_roadlines = (roadlines_coords - center_xy) @ rot_matrix + displacement
        centered_others = (
            XY.reshape(-1, 2) * shift * 3 * raster_size / 512 - center_xy
        ) @ rot_matrix + displacement
        centered_others = centered_others.reshape(128, 11, 2)
        centered_gt = (gt_xy - unscaled_center_xy) @ rot_matrix

        unique_road_ids = np.unique(roadlines_ids)
        for road_id in unique_road_ids:
            if road_id >= 0:
                roadline = centered_roadlines[roadlines_ids == road_id]
                road_type = roadlines_types[roadlines_ids == road_id].flatten()[0]

                road_color = road_colors[road_type]
                for c, rgb in zip(
                    ["green", "yellow", "red"],
                    [(0, 255, 0), (255, 211, 0), (255, 0, 0)],
                ):
                    if road_id in tl_dict[c]:
                        road_color = rgb

                RES_ROADMAP = cv2.polylines(
                    RES_ROADMAP,
                    [roadline.astype(int)],
                    False,
                    road_color,
                    shift=9,
                )

        unique_agent_ids = np.unique(agents_ids)

        is_ego = False
        self_type = 0
        _tmp = 0
        for other_agent_id in unique_agent_ids:
            other_agent_id = int(other_agent_id)
            if other_agent_id < 1:
                continue
            if other_agent_id == agent_id:
                is_ego = True
                self_type = agent_type[agents_ids == other_agent_id]
            else:
                is_ego = False
            _tmp += 1
            agent_lane = centered_others[agents_ids == other_agent_id][0]
            agent_valid = agents_valid[agents_ids == other_agent_id]
            agent_yaw = YAWS[agents_ids == other_agent_id]
            agent_speeds = speeds[agents_ids == other_agent_id]

            agent_l = lengths[agents_ids == other_agent_id]
            agent_w = widths[agents_ids == other_agent_id]
            _type = agent_type[agents_ids == other_agent_id]

            for timestamp, (
                coord,
                valid_coordinate,
                past_yaw,
                agent_speed,
            ) in enumerate(
                zip(
                    agent_lane,
                    agent_valid.flatten(),
                    agent_yaw.flatten(),
                    agent_speeds.flatten(),
                )
            ):
                if valid_coordinate == 0:
                    continue
                box_points = (
                    np.array(
                        [
                            -agent_l,
                            -agent_w,
                            agent_l,
                            -agent_w,
                            agent_l,
                            agent_w,
                            -agent_l,
                            agent_w,
                        ]
                    )
                    .reshape(4, 2)
                    .astype(np.float32)
                    * shift
                    * 3
                    / 2
                    * raster_size
                    / 512
                )

                box_points = (
                    box_points
                    @ np.array(
                        (
                            (np.cos(yaw - past_yaw), -np.sin(yaw - past_yaw)),
                            (np.sin(yaw - past_yaw), np.cos(yaw - past_yaw)),
                        )
                    ).reshape(2, 2)
                )

                _coord = np.array([coord])

                box_points = box_points + _coord
                box_points = box_points.reshape(1, -1, 2).astype(np.int32)

                if is_ego:
                    cv2.fillPoly(
                        RES_EGO[timestamp * 2],
                        box_points,
                        color=int((_type + 1) * 255 / 5),
                        shift=9,
                    )
                    cv2.fillPoly(
                        RES_EGO[timestamp * 2 + 1],
                        box_points,
                        color=int(agent_speed),
                        shift=9,
                    )
                else:
                    cv2.fillPoly(
                        RES_OTHER[timestamp * 2],
                        box_points,
                        color=int((_type + 1) * 255 / 5),
                        shift=9,
                    )
                    cv2.fillPoly(
                        RES_OTHER[timestamp * 2 + 1],
                        box_points,
                        color=int(agent_speed),
                        shift=9,
                    )

        raster = np.concatenate([RES_ROADMAP] + RES_EGO + RES_OTHER, axis=2)

        raster_dict = {
            "object_id": agent_id,
            # "object_ids": agents_ids,
            "raster": raster,
            "yaw": yaw,
            "shift": unscaled_center_xy,
            "_gt_marginal": gt_xy,
            "gt_marginal": centered_gt,
            "future_val_marginal": future_val,
            # "future_val_joint": future_valid[agents_valid.sum(1) > 0],
            "gt_joint": GT_XY[tracks_to_predict.flatten() > 0],
            "future_val_joint": future_valid[tracks_to_predict.flatten() > 0],
            "scenario_id": scenario_id,
            "self_type": self_type,
        }

        GRES.append(raster_dict)

    return GRES


def merge(parsed, validate=False):

    raster_data = rasterize(
        parsed["state/tracks_to_predict"],
        parsed["state/past/x"],
        parsed["state/past/y"],
        parsed["state/current/x"],
        parsed["state/current/y"],
        parsed["state/current/bbox_yaw"],
        parsed["state/past/bbox_yaw"],
        parsed["state/past/valid"],
        parsed["state/current/valid"],
        parsed["state/type"],
        parsed["roadgraph_samples/xyz"],
        parsed["roadgraph_samples/type"],
        parsed["roadgraph_samples/valid"],
        parsed["roadgraph_samples/id"],
        parsed["state/current/width"],
        parsed["state/current/length"],
        parsed["state/id"],
        parsed["traffic_light_state/current/state"],
        parsed["traffic_light_state/current/id"],
        parsed["traffic_light_state/current/valid"],
        parsed["state/future/x"],
        parsed["state/future/y"],
        parsed["state/future/valid"],
        parsed["scenario/id"],
        parsed["state/past/speed"],
        parsed["state/current/speed"],
        validate=validate,
    )

    return raster_data
