import csv
import numpy as np
import os
import shutil
import sys
import copy
from matplotlib import pyplot as plt
import base.parameters_simulation as ps

sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), "tools"))
import traci

from base.driving_env import Highway
from base.parameters_simulation import road_params

# test scenarios
NB_VEHICLES = 25  # 25
speed_range = [15, 25]  # [15, 35]

# Standstill
stop_position_range = [250, 500]  # 300 = 280, 500
stop_speed_range = [0, 10]  # 0
vehicle_distance_range = [12, 22]  # 18
vehicle_start_pos_range = [0, 36]  # 36
vehicles_speed_range = [10, 20]  # 15

# Fast overtaking
vehicles_speeds_range_fast = [10, 20]  # 15
fast_vehicle_speed_range = [15, 60]  # 50
fast_vehicle_start_position_range = [100, 200]  # 150

# test scenarios V2
STOPPED_VEHICLES = 2
FAST_VEHICLES = 0

# test scenarios V3
EVENT_PROBABILITY = 0.05
STOPPED_VEHICLE_EVENT_PROB = 0.5
FREEZE_STEPS = 10
STOP_VEHICLE_RANGE = [100, 200]
FAST_VEHICLE_RANGE = [100, 200]
MIN_STOP_VEHICLE_DISTANCE = 50# 100
MIN_FAST_VEHICLE_DISTANCE = 50# 100
STOP_VEHICLE_STEPS = 10
FAST_VEHICLE_SPEED = 55
FAST_VEHCILE_STEPS = 10


def get_name(filepath):
    return filepath.split("/")[-2]


def ep_log(i, step, episode_reward, nb_safe_actions, nb_hard_safe_actions):
    print(
        f"Episode: {str(i)}\tSteps: {str(step)}\t"
        + f"Reward: {str(episode_reward)}\t"
        + f"Number of safety actions: {str(nb_safe_actions)}\t"
        + f"Number of hard safety actions: {str(nb_hard_safe_actions)}"
    )


def thresh_log(
    thresh,
    episode_rewards,
    episode_steps,
    collissions,
    nb_safe_actions_per_episode,
    nb_safe_hard_actions_per_episode,
):
    print(
        f"\nThreshold: {str(thresh)}\t"
        + f"Mean Reward: {str(np.sum(episode_rewards) / np.sum(episode_steps))}\t"
        + f"Total Steps: {np.sum(episode_steps)}\t"
        + f"Collision Rate: {str(collissions / len(episode_steps) * 100)}%\t"
        + f"Safe action Rate: {np.sum(nb_safe_actions_per_episode)/np.sum(episode_steps)*100}%\t"
        + "Hard action rate: "
        + f"{np.sum(nb_safe_hard_actions_per_episode)/np.sum(episode_steps)*100}%\n\n"
    )


def save_metrics(
    case,
    filepath,
    thresh,
    episode_rewards,
    episode_steps,
    collissions,
    nb_safe_actions_per_episode,
    nb_safe_hard_actions_per_episode,
    collision_speeds,
    stop_events=[],
    fast_events=[],
    mean_speeds=[],
):
    with open(filepath + case + ".csv", "a+") as file:
        writer = csv.writer(file)
        mean_speeds = 0 if len(collision_speeds) == 0 else np.mean(collision_speeds)
        to_write = [
            thresh,
            np.sum(episode_rewards) / np.sum(episode_steps),
            collissions / len(episode_steps),
            np.sum(nb_safe_actions_per_episode) / np.sum(episode_steps),
            np.sum(nb_safe_hard_actions_per_episode) / np.sum(episode_steps),
            mean_speeds,
            np.sum(episode_steps),
        ]
        if len(stop_events) > 0:
            to_write.append(np.mean(stop_events))
        if len(fast_events) > 0:
            to_write.append(np.mean(fast_events))
        if len(mean_speeds) > 0:
            to_write.append(np.mean(mean_speeds))
        writer.writerow(to_write)


def save_uncert(
    case,
    filepath,
    thresh,
    episode,
    uncert,
):
    with open(filepath + case + ".csv", "a+") as file:
        writer = csv.writer(file)
        row = [thresh, episode] + uncert
        writer.writerow(row)


def traci_schema(filepath):
    try:
        traci.gui.setSchema("View #0", "scheme_for_videos")
    except Exception as e:
        print("SUMO scheme not defined.")
    if not os.path.exists("../videos"):
        os.makedirs("../videos")
    name = get_name(filepath)
    video_folder = f"../videos/{name}"
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder)


def traci_before(filepath, case, thresh, episode):
    name = get_name(filepath)
    video_folder = f"../videos/{name}/{case}"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    video_folder = f"{video_folder}/{thresh}_{episode}"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    traci.gui.setZoom("View #0", 8000)
    traci.gui.screenshot("View #0", video_folder + "/init.png")


def traci_each(filepath, case, thresh, episode, step):
    name = get_name(filepath)
    video_folder = f"../videos/{name}/{case}/{thresh}_{episode}/"
    traci.gui.screenshot("View #0", video_folder + str(step) + ".png")


def rerun_test_scenarios(
    dqn,
    filepath,
    ps,
    change_thresh_fn=lambda x: x,
    thresh_range=[0, 1],
    thresh_steps=100,
    use_safe_action=False,
    save_video=False,
    do_save_metrics=True,
    number_tests=1,
    number_episodes=100,
    use_gui=False,
    csv_sufix="",
    do_save_uncert=False,
):
    sufix = "_U" if use_safe_action else "_NU"
    case = "rerun_test_scenarios" + sufix + csv_sufix
    if do_save_metrics or do_save_uncert:
        with open(filepath + case + ".csv", "w+"):
            pass

    ps.sim_params["nb_vehicles"] = NB_VEHICLES
    ps.road_params["speed_range"] = np.array(speed_range)
    env = Highway(
        sim_params=ps.sim_params,
        road_params=ps.road_params,
        use_gui=use_gui,
        return_more_info=True,
    )

    env.reset()
    if save_video:
        traci_schema(filepath)
    range_ = (
        np.linspace(
            thresh_range[0],
            thresh_range[1],
            num=thresh_steps,
        )
        if use_safe_action
        else list(range(number_tests))
    )

    for thresh in range_:
        episode_rewards = []
        episode_steps = []
        nb_safe_actions_per_episode = []
        nb_safe_hard_actions_per_episode = []
        collissions = 0
        collision_speeds = []
        if use_safe_action:
            change_thresh_fn(thresh)
        for i in range(0, number_episodes):
            # np.random.seed(i)
            obs = env.reset()
            if save_video:
                traci_before(filepath, case, thresh, i)
            done = False
            episode_reward = 0
            step = 0
            nb_safe_actions = 0
            nb_hard_safe_actions = 0
            unc = []
            while done is False:
                try:
                    action, action_info = dqn.forward(obs)
                    obs, rewards, done, _, more_info = env.step(action, action_info)
                    reward_no_col = more_info["reward_no_col"]
                    episode_reward += reward_no_col
                    step += 1
                    if more_info["ego_collision"]:
                        collissions += 1
                        collision_speeds.append(more_info["ego_speed"])
                    if "safe_action" in action_info:
                        nb_safe_actions += action_info["safe_action"]
                        nb_hard_safe_actions += action_info["hard_safe"]
                    if save_video:
                        traci_each(filepath, case, thresh, i, step)
                    if "coefficient_of_variation" in action_info:
                        unc.append(action_info["coefficient_of_variation"][action])
                except Exception as e:
                    print(e)
                    done = False
                    episode_reward = 0
                    step = 0
                    nb_safe_actions = 0
                    nb_hard_safe_actions = 0
                    unc = []
                    env = Highway(
                        sim_params=ps.sim_params,
                        road_params=ps.road_params,
                        use_gui=use_gui,
                        return_more_info=True,
                    )

                    obs = env.reset()
                    if save_video:
                        traci_before(filepath, case, thresh, i)

            if do_save_uncert:
                save_uncert(case, filepath, thresh, i, unc)
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            nb_safe_actions_per_episode.append(nb_safe_actions)
            nb_safe_hard_actions_per_episode.append(nb_hard_safe_actions)
            ep_log(i, step, episode_reward, nb_safe_actions, nb_hard_safe_actions)
        thresh_log(
            thresh,
            episode_rewards,
            episode_steps,
            collissions,
            nb_safe_actions_per_episode,
            nb_safe_hard_actions_per_episode,
        )
        if do_save_metrics:
            save_metrics(
                case,
                filepath,
                thresh,
                episode_rewards,
                episode_steps,
                collissions,
                nb_safe_actions_per_episode,
                nb_safe_hard_actions_per_episode,
                collision_speeds,
            )
    env.close()


def fast_overtaking(
    dqn,
    filepath,
    ps,
    change_thresh_fn=lambda x: x,
    thresh_range=[0, 1],
    thresh_steps=100,
    use_safe_action=False,
    save_video=False,
    do_save_metrics=True,
    number_tests=1,
    number_episodes=100,
    use_gui=False,
    csv_sufix="",
    do_save_uncert=False,
):
    sufix = "_U" if use_safe_action else "_NU"
    case = "fast_overtaking" + sufix + csv_sufix
    if do_save_metrics or do_save_uncert:
        with open(filepath + case + ".csv", "w+"):
            pass
    ps.sim_params["nb_vehicles"] = 5
    env = Highway(
        sim_params=ps.sim_params,
        road_params=ps.road_params,
        use_gui=use_gui,
        return_more_info=True,
    )

    env.reset()

    if save_video:
        traci_schema(filepath)
    range_ = (
        np.linspace(
            thresh_range[0],
            thresh_range[1],
            num=thresh_steps,
        )
        if use_safe_action
        else list(range(number_tests))
    )
    for thresh in range_:
        episode_rewards = []
        episode_steps = []
        nb_safe_actions_per_episode = []
        nb_safe_hard_actions_per_episode = []
        collissions = 0
        collision_speeds = []
        if use_safe_action:
            change_thresh_fn(thresh)
        for i in range(0, number_episodes):
            vehicles_speeds_fast = np.random.uniform(
                vehicles_speeds_range_fast[0],
                vehicles_speeds_range_fast[1],
            )
            fast_vehicle_speed = np.random.uniform(
                fast_vehicle_speed_range[0],
                fast_vehicle_speed_range[1],
            )
            fast_vehicle_start_position = np.random.uniform(
                fast_vehicle_start_position_range[0],
                fast_vehicle_start_position_range[1],
            )
            # Make sure that the vehicles are not affected by previous state
            # np.random.seed(57)
            def set_fast_overtaking():
                env.reset()
                s0 = 1000.0
                traci.vehicle.moveTo("veh0", "highway_0", s0 - 300)
                traci.vehicle.moveTo("veh1", "highway_1", s0 - 300)
                traci.vehicle.moveTo("veh2", "highway_2", s0 - 300)
                traci.vehicle.moveTo("veh3", "highway_2", s0 - 400)
                traci.vehicle.moveTo("veh4", "highway_2", s0 - 500)
                traci.simulationStep()
                env.speeds[0, 0] = 15
                for veh in env.vehicles:
                    traci.vehicle.setSpeedMode(veh, 0)
                traci.vehicle.setSpeed("veh0", 15)
                traci.vehicle.setSpeed("veh1", 15)
                traci.vehicle.setSpeed("veh2", 55)
                traci.vehicle.setSpeed("veh3", 15)
                traci.vehicle.setSpeed("veh4", 15)
                traci.vehicle.setMaxSpeed("veh0", 25)
                traci.vehicle.setMaxSpeed("veh1", 15)
                traci.vehicle.setMaxSpeed("veh2", 55)
                traci.vehicle.setMaxSpeed("veh3", 15)
                traci.vehicle.setMaxSpeed("veh4", 15)
                traci.simulationStep()
                env.step(0)

                # Overtaking case
                traci.vehicle.moveTo("veh0", "highway_0", s0)
                traci.vehicle.moveTo("veh1", "highway_0", s0 + 50)
                traci.vehicle.moveTo(
                    "veh2", "highway_1", s0 - fast_vehicle_start_position
                )
                traci.vehicle.moveTo("veh3", "highway_2", s0 - 50)
                traci.vehicle.moveTo("veh4", "highway_2", s0 - 0)
                traci.vehicle.setSpeed("veh0", vehicles_speeds_fast)
                traci.vehicle.setSpeed("veh1", vehicles_speeds_fast)
                traci.vehicle.setSpeed("veh2", fast_vehicle_speed)
                traci.vehicle.setSpeed("veh3", vehicles_speeds_fast)
                traci.vehicle.setSpeed("veh4", vehicles_speeds_fast)
                observation, reward, _, _, more_info = env.step(0)
                for veh in env.vehicles[1:]:
                    traci.vehicle.setSpeed(veh, -1)
                return observation, reward, more_info

            try:
                observation, reward, more_info = set_fast_overtaking()
            except Exception as e:
                print(e)
                observation, reward, more_info = set_fast_overtaking()

            # Run fast overtaking case
            if save_video:
                traci_before(filepath, case, thresh, i)
            action_log = []
            nb_safe_actions = 0
            nb_hard_safe_actions = 0
            episode_reward = 0
            step = 0
            unc = []
            while step < 8:
                try:
                    action, action_info = dqn.forward(observation)
                    observation, reward, done, _, more_info = env.step(
                        action, action_info
                    )
                    episode_reward += reward
                    step += 1
                    if more_info["ego_collision"]:
                        collissions += 1
                        collision_speeds.append(more_info["ego_speed"])
                    if "safe_action" in action_info:
                        nb_safe_actions += action_info["safe_action"]
                        nb_hard_safe_actions += action_info["hard_safe"]
                    action_log.append(action)
                    if save_video:
                        traci_each(filepath, case, thresh, i, step)
                    if "coefficient_of_variation" in action_info:
                        unc.append(action_info["coefficient_of_variation"][action])
                    if done:
                        break
                except Exception as e:
                    print(e)
                    env = Highway(
                        sim_params=ps.sim_params,
                        road_params=ps.road_params,
                        use_gui=use_gui,
                        return_more_info=True,
                    )

                    env.reset()
                    observation, reward, more_info = set_fast_overtaking()

                    if save_video:
                        traci_before(filepath, case, thresh, i)
                    action_log = []
                    nb_safe_actions = 0
                    nb_hard_safe_actions = 0
                    episode_reward = 0
                    step = 0
                    unc = []

            if do_save_uncert:
                save_uncert(case, filepath, thresh, i, unc)
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            nb_safe_actions_per_episode.append(nb_safe_actions)
            nb_safe_hard_actions_per_episode.append(nb_hard_safe_actions)
            ep_log(i, step, episode_reward, nb_safe_actions, nb_hard_safe_actions)
        thresh_log(
            thresh,
            episode_rewards,
            episode_steps,
            collissions,
            nb_safe_actions_per_episode,
            nb_safe_hard_actions_per_episode,
        )
        if do_save_metrics:
            save_metrics(
                case,
                filepath,
                thresh,
                episode_rewards,
                episode_steps,
                collissions,
                nb_safe_actions_per_episode,
                nb_safe_hard_actions_per_episode,
                collision_speeds,
            )
    env.close()


def standstill(
    dqn,
    filepath,
    ps,
    change_thresh_fn=lambda x: x,
    thresh_range=[0, 1],
    thresh_steps=100,
    use_safe_action=False,
    save_video=False,
    do_save_metrics=True,
    number_tests=1,
    number_episodes=100,
    use_gui=False,
    csv_sufix="",
    do_save_uncert=False,
):
    sufix = "_U" if use_safe_action else "_NU"
    case = "standstill" + sufix + csv_sufix
    if do_save_metrics or do_save_uncert:
        with open(filepath + case + ".csv", "w+"):
            pass
    ps.sim_params["nb_vehicles"] = 7
    env = Highway(
        sim_params=ps.sim_params,
        road_params=ps.road_params,
        use_gui=use_gui,
        return_more_info=True,
    )

    env.reset()
    if save_video:
        traci_schema(filepath)
    range_ = (
        np.linspace(
            thresh_range[0],
            thresh_range[1],
            num=thresh_steps,
        )
        if use_safe_action
        else list(range(number_tests))
    )
    for thresh in range_:
        episode_rewards = []
        episode_steps = []
        nb_safe_actions_per_episode = []
        nb_safe_hard_actions_per_episode = []
        collision_speeds = []
        collissions = 0
        if use_safe_action:
            change_thresh_fn(thresh)
        for i in range(0, number_episodes):
            # Make sure that the vehicles are not affected by previous state
            # np.random.seed(57)
            def set_standstill():
                env.reset()
                s0 = 1000.0
                stop_speed = np.random.uniform(
                    stop_speed_range[0],
                    stop_speed_range[1],
                )
                vehicle_distance = np.random.uniform(
                    vehicle_distance_range[0], vehicle_distance_range[1]
                )
                vehicle_start_pos = np.random.uniform(
                    vehicle_start_pos_range[0], vehicle_start_pos_range[1]
                )
                vehicles_speed = np.random.uniform(
                    vehicles_speed_range[0], vehicles_speed_range[1]
                )
                traci.vehicle.moveTo("veh0", "highway_0", s0 - 300)
                traci.vehicle.moveTo("veh1", "highway_1", s0 - 300)
                traci.vehicle.moveTo("veh2", "highway_2", s0 - 300)
                traci.vehicle.moveTo("veh3", "highway_2", s0 - 400)
                traci.vehicle.moveTo("veh4", "highway_2", s0 - 500)
                traci.vehicle.moveTo("veh5", "highway_2", s0 - 600)
                traci.vehicle.moveTo("veh6", "highway_2", s0 - 700)
                traci.simulationStep()
                env.speeds[0, 0] = 25
                for veh in env.vehicles:
                    traci.vehicle.setSpeedMode(veh, 0)
                    traci.vehicle.setLaneChangeMode(veh, 0)  # Turn off all lane changes
                traci.vehicle.setSpeed("veh0", 25)
                traci.vehicle.setSpeed("veh1", stop_speed)
                traci.vehicle.setSpeed("veh2", vehicles_speed)
                traci.vehicle.setSpeed("veh3", vehicles_speed)
                traci.vehicle.setSpeed("veh4", vehicles_speed)
                traci.vehicle.setSpeed("veh5", vehicles_speed)
                traci.vehicle.setSpeed("veh6", vehicles_speed)
                traci.simulationStep()
                traci.vehicle.setMaxSpeed("veh0", 25)
                traci.vehicle.setMaxSpeed("veh1", stop_speed)
                traci.vehicle.setMaxSpeed("veh2", vehicles_speed)
                traci.vehicle.setMaxSpeed("veh3", vehicles_speed)
                traci.vehicle.setMaxSpeed("veh4", vehicles_speed)
                traci.vehicle.setMaxSpeed("veh5", vehicles_speed)
                traci.vehicle.setMaxSpeed("veh6", vehicles_speed)
                traci.simulationStep()
                env.step(0)

                # Standstill case
                traci.vehicle.moveTo("veh0", "highway_0", s0)
                traci.vehicle.moveTo(
                    "veh1",
                    "highway_0",
                    s0
                    + np.random.uniform(stop_position_range[0], stop_position_range[1]),
                )
                traci.vehicle.moveTo("veh2", "highway_1", s0 + vehicle_start_pos)
                traci.vehicle.moveTo(
                    "veh3", "highway_1", s0 + vehicle_start_pos + vehicle_distance
                )
                traci.vehicle.moveTo(
                    "veh4", "highway_1", s0 + vehicle_start_pos + 2 * vehicle_distance
                )
                traci.vehicle.moveTo(
                    "veh5", "highway_1", s0 + vehicle_start_pos + 3 * vehicle_distance
                )
                traci.vehicle.moveTo(
                    "veh6", "highway_1", s0 + vehicle_start_pos + 4 * vehicle_distance
                )
                traci.vehicle.setSpeed("veh0", 25)
                traci.vehicle.setSpeed("veh1", stop_speed)
                traci.vehicle.setSpeed("veh2", vehicles_speed)
                traci.vehicle.setSpeed("veh3", vehicles_speed)
                traci.vehicle.setSpeed("veh4", vehicles_speed)
                traci.vehicle.setSpeed("veh5", vehicles_speed)
                traci.vehicle.setSpeed("veh6", vehicles_speed)
                observation, reward, _, _, more_info = env.step(0)
                for veh in env.vehicles[1:]:
                    traci.vehicle.setSpeed(veh, -1)
                return observation, reward, more_info

            try:
                observation, reward, more_info = set_standstill()
            except Exception as e:
                print(e)
                observation, reward, more_info = set_standstill()

            # Run standstill case
            if save_video:
                traci_before(filepath, case, thresh, i)
            nb_safe_actions = 0
            nb_hard_safe_actions = 0

            episode_reward = 0
            unc = []
            step = 0
            while step < 15:
                try:
                    action, action_info = dqn.forward(observation)
                    observation, reward, done, _, more_info = env.step(action)
                    episode_reward += reward
                    step += 1
                    if more_info["ego_collision"]:
                        collissions += 1
                        collision_speeds.append(more_info["ego_speed"])
                    if "safe_action" in action_info:
                        nb_safe_actions += action_info["safe_action"]
                        nb_hard_safe_actions += action_info["hard_safe"]
                    if save_video:
                        traci_each(filepath, case, thresh, i, step)
                    if "coefficient_of_variation" in action_info:
                        unc.append(action_info["coefficient_of_variation"][action])
                    if done:
                        break
                except Exception as e:
                    print(e)
                    env = Highway(
                        sim_params=ps.sim_params,
                        road_params=ps.road_params,
                        use_gui=use_gui,
                        return_more_info=True,
                    )

                    env.reset()
                    observation, reward, more_info = set_standstill()
                    if save_video:
                        traci_before(filepath, case, thresh, i)
                    nb_safe_actions = 0
                    nb_hard_safe_actions = 0

                    episode_reward = 0
                    unc = []
                    step = 0

            if do_save_uncert:
                save_uncert(case, filepath, thresh, i, unc)
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            nb_safe_actions_per_episode.append(nb_safe_actions)
            nb_safe_hard_actions_per_episode.append(nb_hard_safe_actions)
            ep_log(i, step, episode_reward, nb_safe_actions, nb_hard_safe_actions)
        thresh_log(
            thresh,
            episode_rewards,
            episode_steps,
            collissions,
            nb_safe_actions_per_episode,
            nb_safe_hard_actions_per_episode,
        )
        if do_save_metrics:
            save_metrics(
                case,
                filepath,
                thresh,
                episode_rewards,
                episode_steps,
                collissions,
                nb_safe_actions_per_episode,
                nb_safe_hard_actions_per_episode,
                collision_speeds,
            )
    env.close()


def standstill_v2(
    dqn,
    filepath,
    ps,
    use_safe_action=False,
    save_video=False,
    position_steps=100,
    use_gui=False,
    csv_sufix="_v2",
    do_save_uncert=False,
):
    sufix = "_U" if use_safe_action else "_NU"
    case = "standstill" + sufix + csv_sufix
    if do_save_uncert:
        with open(filepath + case + ".csv", "w+"):
            pass
    ps.sim_params["nb_vehicles"] = 7
    env = Highway(
        sim_params=ps.sim_params,
        road_params=ps.road_params,
        use_gui=use_gui,
        return_more_info=True,
    )

    env.reset()
    if save_video:
        traci_schema(filepath)
    range_ = np.linspace(
        stop_position_range[0],
        stop_position_range[1],
        num=position_steps,
    )
    for position in range_:
        # Make sure that the vehicles are not affected by previous state
        np.random.seed(57)

        def set_standstill():
            env.reset()
            if use_gui:
                traci.vehicle.setColor("veh1", (255, 255, 255))
            s0 = 1000.0
            traci.vehicle.moveTo("veh0", "highway_0", s0 - 300)
            traci.vehicle.moveTo("veh1", "highway_1", s0 - 300)
            traci.vehicle.moveTo("veh2", "highway_2", s0 - 300)
            traci.vehicle.moveTo("veh3", "highway_2", s0 - 400)
            traci.vehicle.moveTo("veh4", "highway_2", s0 - 500)
            traci.vehicle.moveTo("veh5", "highway_2", s0 - 600)
            traci.vehicle.moveTo("veh6", "highway_2", s0 - 700)
            traci.simulationStep()
            env.speeds[0, 0] = 25
            for veh in env.vehicles:
                traci.vehicle.setSpeedMode(veh, 0)
                traci.vehicle.setLaneChangeMode(veh, 0)  # Turn off all lane changes
            traci.vehicle.setSpeed("veh0", 25)
            traci.vehicle.setSpeed("veh1", 0)
            traci.vehicle.setSpeed("veh2", 15)
            traci.vehicle.setSpeed("veh3", 15)
            traci.vehicle.setSpeed("veh4", 15)
            traci.vehicle.setSpeed("veh5", 15)
            traci.vehicle.setSpeed("veh6", 15)
            traci.simulationStep()
            traci.vehicle.setMaxSpeed("veh0", 25)
            traci.vehicle.setMaxSpeed("veh1", 0.001)
            traci.vehicle.setMaxSpeed("veh2", 15)
            traci.vehicle.setMaxSpeed("veh3", 15)
            traci.vehicle.setMaxSpeed("veh4", 15)
            traci.vehicle.setMaxSpeed("veh5", 15)
            traci.vehicle.setMaxSpeed("veh6", 15)
            traci.simulationStep()
            env.step(0)
            if use_gui:
                traci.vehicle.setColor("veh1", (255, 255, 255))

            # Standstill case
            traci.vehicle.moveTo("veh0", "highway_0", s0)
            traci.vehicle.moveTo(
                "veh1",
                "highway_0",
                s0 + position,
            )
            traci.vehicle.moveTo("veh2", "highway_1", s0 + 36)
            traci.vehicle.moveTo("veh3", "highway_1", s0 + 54)
            traci.vehicle.moveTo("veh4", "highway_1", s0 + 72)
            traci.vehicle.moveTo("veh5", "highway_1", s0 + 90)
            traci.vehicle.moveTo("veh6", "highway_1", s0 + 108)
            traci.vehicle.setSpeed("veh0", 25)
            traci.vehicle.setSpeed("veh1", 0)
            traci.vehicle.setSpeed("veh2", 15)
            traci.vehicle.setSpeed("veh3", 15)
            traci.vehicle.setSpeed("veh4", 15)
            traci.vehicle.setSpeed("veh5", 15)
            traci.vehicle.setSpeed("veh6", 15)
            observation, reward, _, _, more_info = env.step(0)
            for veh in env.vehicles[1:]:
                traci.vehicle.setSpeed(veh, -1)
            return observation, reward, more_info

        try:
            observation, reward, more_info = set_standstill()
        except Exception as e:
            print(e)
            observation, reward, more_info = set_standstill()

        # Run standstill case
        if save_video:
            traci_before(filepath, case, 0, position)
        nb_safe_actions = 0
        nb_hard_safe_actions = 0

        episode_reward = 0
        unc = []
        step = 0
        while step < 30:
            try:
                action, action_info = dqn.forward(observation)
                observation, reward, done, _, more_info = env.step(action)
                episode_reward += reward
                step += 1
                if "safe_action" in action_info:
                    nb_safe_actions += action_info["safe_action"]
                    nb_hard_safe_actions += action_info["hard_safe"]
                if save_video:
                    traci_each(filepath, case, 0, position, step)
                if "coefficient_of_variation" in action_info:
                    unc.append(action_info["coefficient_of_variation"][action])
                if done:
                    break
            except Exception as e:
                print(e)
                env = Highway(
                    sim_params=ps.sim_params,
                    road_params=ps.road_params,
                    use_gui=use_gui,
                    return_more_info=True,
                )

                env.reset()
                observation, reward, more_info = set_standstill()
                if save_video:
                    traci_before(filepath, case, 0, position)
                nb_safe_actions = 0
                nb_hard_safe_actions = 0

                episode_reward = 0
                unc = []
                step = 0

        if do_save_uncert:
            save_uncert(case, filepath, 0, position, unc)
        ep_log(0, step, episode_reward, nb_safe_actions, nb_hard_safe_actions)
    env.close()


def fast_overtaking_v2(
    dqn,
    filepath,
    ps,
    use_safe_action=False,
    save_video=False,
    position_steps=1,
    use_gui=False,
    csv_sufix="_v2",
    do_save_uncert=False,
):
    sufix = "_U" if use_safe_action else "_NU"
    case = "fast_overtaking" + sufix + csv_sufix
    if do_save_uncert:
        with open(filepath + case + ".csv", "w+"):
            pass
    ps.sim_params["nb_vehicles"] = 5
    env = Highway(
        sim_params=ps.sim_params,
        road_params=ps.road_params,
        use_gui=use_gui,
        return_more_info=True,
    )

    env.reset()

    if save_video:
        traci_schema(filepath)
    range_ = np.linspace(
        fast_vehicle_speed_range[0],
        fast_vehicle_speed_range[1],
        num=position_steps,
    )
    for fast_vehicle_speed in range_:
        # Make sure that the vehicles are not affected by previous state
        np.random.seed(57)

        def set_fast_overtaking():
            env.reset()
            s0 = 1000.0
            traci.vehicle.moveTo("veh0", "highway_0", s0 - 300)
            traci.vehicle.moveTo("veh1", "highway_1", s0 - 300)
            traci.vehicle.moveTo("veh2", "highway_2", s0 - 300)
            traci.vehicle.moveTo("veh3", "highway_2", s0 - 400)
            traci.vehicle.moveTo("veh4", "highway_2", s0 - 500)
            traci.simulationStep()
            env.speeds[0, 0] = 15
            for veh in env.vehicles:
                traci.vehicle.setSpeedMode(veh, 0)
            traci.vehicle.setSpeed("veh0", 15)
            traci.vehicle.setSpeed("veh1", 15)
            traci.vehicle.setSpeed("veh2", 55)
            traci.vehicle.setSpeed("veh3", 15)
            traci.vehicle.setSpeed("veh4", 15)
            traci.vehicle.setMaxSpeed("veh0", 25)
            traci.vehicle.setMaxSpeed("veh1", 15)
            traci.vehicle.setMaxSpeed("veh2", 55)
            traci.vehicle.setMaxSpeed("veh3", 15)
            traci.vehicle.setMaxSpeed("veh4", 15)
            traci.simulationStep()
            env.step(0)

            # Overtaking case
            traci.vehicle.moveTo("veh0", "highway_0", s0)
            traci.vehicle.moveTo("veh1", "highway_0", s0 + 50)
            traci.vehicle.moveTo("veh2", "highway_1", s0 - 150)
            traci.vehicle.moveTo("veh3", "highway_2", s0 - 50)
            traci.vehicle.moveTo("veh4", "highway_2", s0 - 0)
            traci.vehicle.setSpeed("veh0", 15)
            traci.vehicle.setSpeed("veh1", 15)
            traci.vehicle.setSpeed("veh2", fast_vehicle_speed)
            traci.vehicle.setSpeed("veh3", 15)
            traci.vehicle.setSpeed("veh4", 15)
            observation, reward, _, _, more_info = env.step(0)
            for veh in env.vehicles[1:]:
                traci.vehicle.setSpeed(veh, -1)
            return observation, reward, more_info

        try:
            observation, reward, more_info = set_fast_overtaking()
        except Exception as e:
            print(e)
            observation, reward, more_info = set_fast_overtaking()

        # Run fast overtaking case
        if save_video:
            traci_before(filepath, case, 0, fast_vehicle_speed)
        action_log = []
        nb_safe_actions = 0
        nb_hard_safe_actions = 0
        episode_reward = 0
        step = 0
        unc = []
        while step < 16:
            try:
                action, action_info = dqn.forward(observation)
                observation, reward, done, _, more_info = env.step(action, action_info)
                episode_reward += reward
                step += 1
                if "safe_action" in action_info:
                    nb_safe_actions += action_info["safe_action"]
                    nb_hard_safe_actions += action_info["hard_safe"]
                action_log.append(action)
                if save_video:
                    traci_each(filepath, case, 0, fast_vehicle_speed, step)
                if "coefficient_of_variation" in action_info:
                    unc.append(action_info["coefficient_of_variation"][action])
                if done:
                    break
            except Exception as e:
                print(e)
                env = Highway(
                    sim_params=ps.sim_params,
                    road_params=ps.road_params,
                    use_gui=use_gui,
                    return_more_info=True,
                )

                env.reset()
                observation, reward, more_info = set_fast_overtaking()

                if save_video:
                    traci_before(filepath, case, 0, fast_vehicle_speed)
                action_log = []
                nb_safe_actions = 0
                nb_hard_safe_actions = 0
                episode_reward = 0
                step = 0
                unc = []

        if do_save_uncert:
            save_uncert(case, filepath, 0, fast_vehicle_speed, unc)
        ep_log(0, step, episode_reward, nb_safe_actions, nb_hard_safe_actions)
    env.close()


def rerun_test_scenarios_v2(
    dqn,
    filepath,
    ps,
    change_thresh_fn=lambda x: x,
    thresh_range=np.linspace(0, 1, 10),
    use_safe_action=False,
    save_video=False,
    do_save_metrics=True,
    number_tests=1,
    number_episodes=100,
    use_gui=False,
    csv_sufix="",
    do_save_uncert=False,
):
    sufix = "_U" if use_safe_action else "_NU"
    case = "rerun_test_scenarios" + sufix + csv_sufix
    if do_save_metrics or do_save_uncert:
        with open(filepath + case + ".csv", "w+"):
            pass

    env = Highway(
        sim_params=ps.sim_params,
        road_params=ps.road_params,
        use_gui=use_gui,
        return_more_info=True,
    )

    env.reset()
    if save_video:
        traci_schema(filepath)
    range_ = thresh_range if use_safe_action else list(range(number_tests))

    for thresh in range_:
        episode_rewards = []
        episode_steps = []
        nb_safe_actions_per_episode = []
        nb_safe_hard_actions_per_episode = []
        collissions = 0
        collision_speeds = []
        if use_safe_action:
            change_thresh_fn(thresh)
        for i in range(0, number_episodes):
            # np.random.seed(i)
            obs = env.reset()

            vehicles = traci.vehicle.getIDList()[1:]
            chosen_vehicles = np.random.choice(
                vehicles, size=(STOPPED_VEHICLES + FAST_VEHICLES), replace=False
            )
            chosen_stop_vehicles = chosen_vehicles[:STOPPED_VEHICLES]
            for v in chosen_stop_vehicles:
                traci.vehicle.setSpeed(v, 0)
                traci.vehicle.setMaxSpeed(v, 0.001)

            chosen_fast_vehicles = chosen_vehicles[STOPPED_VEHICLES:]
            for v in chosen_fast_vehicles:
                traci.vehicle.setSpeed(v, 55)
                traci.vehicle.setMaxSpeed(v, 55)
                pos = np.random.uniform(low=0, high=500)
                traci.vehicle.moveTo(v, "highway_0", -pos)

            if save_video:
                traci_before(filepath, case, thresh, i)
            done = False
            episode_reward = 0
            step = 0
            nb_safe_actions = 0
            nb_hard_safe_actions = 0
            unc = []
            while done is False:
                try:
                    action, action_info = dqn.forward(obs)
                    obs, rewards, done, _, more_info = env.step(action, action_info)
                    reward_no_col = more_info["reward_no_col"]
                    episode_reward += reward_no_col
                    step += 1
                    if more_info["ego_collision"]:
                        collissions += 1
                        collision_speeds.append(more_info["ego_speed"])
                    if "safe_action" in action_info:
                        nb_safe_actions += action_info["safe_action"]
                        nb_hard_safe_actions += action_info["hard_safe"]
                    if save_video:
                        traci_each(filepath, case, thresh, i, step)
                    if "coefficient_of_variation" in action_info:
                        unc.append(action_info["coefficient_of_variation"][action])
                except Exception as e:
                    print(e)
                    done = False
                    episode_reward = 0
                    step = 0
                    nb_safe_actions = 0
                    nb_hard_safe_actions = 0
                    unc = []
                    env = Highway(
                        sim_params=ps.sim_params,
                        road_params=ps.road_params,
                        use_gui=use_gui,
                        return_more_info=True,
                    )

                    obs = env.reset()
                    if save_video:
                        traci_before(filepath, case, thresh, i)

            if do_save_uncert:
                save_uncert(case, filepath, thresh, i, unc)
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            nb_safe_actions_per_episode.append(nb_safe_actions)
            nb_safe_hard_actions_per_episode.append(nb_hard_safe_actions)
            ep_log(i, step, episode_reward, nb_safe_actions, nb_hard_safe_actions)
        thresh_log(
            thresh,
            episode_rewards,
            episode_steps,
            collissions,
            nb_safe_actions_per_episode,
            nb_safe_hard_actions_per_episode,
        )
        if do_save_metrics:
            save_metrics(
                case,
                filepath,
                thresh,
                episode_rewards,
                episode_steps,
                collissions,
                nb_safe_actions_per_episode,
                nb_safe_hard_actions_per_episode,
                collision_speeds,
            )
    env.close()


def rerun_test_scenarios_v0(
    dqn,
    filepath,
    ps,
    change_thresh_fn=lambda x: x,
    thresh_range=np.linspace(0, 1, 10),
    use_safe_action=False,
    save_video=False,
    do_save_metrics=True,
    number_tests=1,
    number_episodes=100,
    use_gui=False,
    csv_sufix="_v0",
    do_save_uncert=False,
):
    sufix = "_U" if use_safe_action else "_NU"
    case = "rerun_test_scenarios" + sufix + csv_sufix
    if do_save_metrics or do_save_uncert:
        with open(filepath + case + ".csv", "w+"):
            pass

    env = Highway(
        sim_params=ps.sim_params,
        road_params=ps.road_params,
        use_gui=use_gui,
        return_more_info=True,
    )

    env.reset()
    if save_video:
        traci_schema(filepath)
    range_ = thresh_range if use_safe_action else list(range(number_tests))

    for thresh in range_:
        episode_rewards = []
        episode_steps = []
        nb_safe_actions_per_episode = []
        nb_safe_hard_actions_per_episode = []
        collissions = 0
        collision_speeds = []
        if use_safe_action:
            change_thresh_fn(thresh)
        for i in range(0, number_episodes):
            # np.random.seed(i)
            obs = env.reset()
            if save_video:
                traci_before(filepath, case, thresh, i)
            done = False
            episode_reward = 0
            step = 0
            nb_safe_actions = 0
            nb_hard_safe_actions = 0
            unc = []
            while done is False:
                action, action_info = dqn.forward(obs)
                obs, rewards, done, _, more_info = env.step(action, action_info)
                reward_no_col = more_info["reward_no_col"]
                episode_reward += reward_no_col
                step += 1
                if more_info["ego_collision"]:
                    collissions += 1
                    collision_speeds.append(more_info["ego_speed"])
                if "safe_action" in action_info:
                    nb_safe_actions += action_info["safe_action"]
                    nb_hard_safe_actions += action_info["hard_safe"]
                if save_video:
                    traci_each(filepath, case, thresh, i, step)
                if "coefficient_of_variation" in action_info:
                    unc.append(action_info["coefficient_of_variation"][action])


            if do_save_uncert:
                save_uncert(case, filepath, thresh, i, unc)
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            nb_safe_actions_per_episode.append(nb_safe_actions)
            nb_safe_hard_actions_per_episode.append(nb_hard_safe_actions)
            ep_log(i, step, episode_reward, nb_safe_actions, nb_hard_safe_actions)
        thresh_log(
            thresh,
            episode_rewards,
            episode_steps,
            collissions,
            nb_safe_actions_per_episode,
            nb_safe_hard_actions_per_episode,
        )
        if do_save_metrics:
            save_metrics(
                case,
                filepath,
                thresh,
                episode_rewards,
                episode_steps,
                collissions,
                nb_safe_actions_per_episode,
                nb_safe_hard_actions_per_episode,
                collision_speeds,
            )
    env.close()


stopped_vh = {
    "id": -1,
    "speed": -1,
    "max_speed": road_params['speed_range'][1],
    "min_speed": road_params['speed_range'][0],
}


def resume_stop_vehicle():
    vh = stopped_vh["id"]
    speed = stopped_vh["min_speed"]
    max_speed = stopped_vh["max_speed"]
    try:
        traci.vehicle.setSpeed(vh, speed)
        traci.vehicle.setMaxSpeed(vh, max_speed)
    except:
        print("Vehicle does not exist")


def stop_vehicle():
    id_list = traci.vehicle.getIDList()

    # agent_lane = traci.vehicle.getLaneIndex(id_list[0])
    agent_pos = traci.vehicle.getPosition(id_list[0])[0]

    vehicles = id_list[1:]
    distances = []
    ids = []
    for vh in vehicles:
        # vh_lane = traci.vehicle.getLaneIndex(vh)
        vh_pos = traci.vehicle.getPosition(vh)[0]
        distance = vh_pos - agent_pos
        if (
            # vh_lane == agent_lane
            # and 
            STOP_VEHICLE_RANGE[0] < distance < STOP_VEHICLE_RANGE[1]
        ):
            ids.append(vh)
            distances.append(distance)

    if len(ids) == 0:
        print("Vehicle not selected")
        return False
    else:
        idcs = np.argsort(distances)
        selected_id = ids[idcs[0]]

        stopped_vh["id"] = copy.copy(selected_id)
        stopped_vh["speed"] = traci.vehicle.getSpeed(selected_id)
        # stopped_vh["max_speed"] = traci.vehicle.getMaxSpeed(selected_vh["id"])
        traci.vehicle.setSpeed(selected_id, 0)
        traci.vehicle.setMaxSpeed(selected_id, 0.001)
    return True


fast_vh = {
    "id": -1,
    "speed": -1,
    "max_speed": road_params['speed_range'][1],
    "min_speed": road_params['speed_range'][0],
}


def resume_fast_vehicle():
    vh = fast_vh["id"]
    speed = fast_vh["min_speed"]
    max_speed = fast_vh["max_speed"]
    try:
        traci.vehicle.setMaxSpeed(vh, max_speed)
        traci.vehicle.setSpeed(vh, speed)
    except:
        print("Vehicle does not exist")


def fast_vehicle():
    id_list = traci.vehicle.getIDList()

    # agent_lane = traci.vehicle.getLaneIndex(id_list[0])
    agent_pos = traci.vehicle.getPosition(id_list[0])[0]

    vehicles = id_list[1:]
    distances = []
    ids = []
    for vh in vehicles:
        # vh_lane = traci.vehicle.getLaneIndex(vh)
        vh_pos = traci.vehicle.getPosition(vh)[0]
        distance = agent_pos - vh_pos
        if (
            # vh_lane != agent_lane
            # and 
            FAST_VEHICLE_RANGE[0] < distance < FAST_VEHICLE_RANGE[1]
        ):
            ids.append(vh)
            distances.append(distance)

    if len(ids) == 0:
        print("Vehicle not selected")
        return False
    else:
        idcs = np.argsort(distances)
        selected_id = ids[idcs[0]]

        fast_vh["id"] = copy.copy(selected_id)
        fast_vh["speed"] = traci.vehicle.getSpeed(selected_id)
        # fast_vh["max_speed"] = traci.vehicle.getMaxSpeed(selected_id)
        traci.vehicle.setSpeed(selected_id, FAST_VEHICLE_SPEED)
        traci.vehicle.setMaxSpeed(selected_id, FAST_VEHICLE_SPEED)
    return True

def plot_metrics_hist(action_info, filepath, case, thresh, episode, step):
    name = get_name(filepath)
    plot_path = f"../videos/{name}/{case}/{thresh}_{episode}/plot_{str(step)}.png"

    if "var" in action_info and "mse" in action_info:
        var, mse = action_info["var"], action_info["mse"]
        # print(f'Max VAR: {np.max(var)}\tIndex VAR: {np.argmax(var)}\tMax MSE: {np.mean(mse)}\tIndex MSE: {np.argmax(mse)}')
        plt.figure()
        fig, axs = plt.subplots(ncols=2, nrows=1)
        fig.set_figwidth(10)
        fig.set_figheight(5)
        axs[0].hist(var, bins=30)#, range=(0, 100)
        axs[1].hist(mse, bins=30)#, range=(0, 100)
        axs[0].set_title("Sigma Distribution", fontsize=25)
        axs[1].set_title("MSE Distribution", fontsize=25)
        fig.savefig(plot_path)
        plt.close()


def rerun_test_scenarios_v3(
    dqn,
    filepath,
    ps,
    change_thresh_fn=lambda x: x,
    thresh_range=np.linspace(0, 1, 10),
    use_safe_action=False,
    save_video=False,
    do_save_metrics=True,
    number_tests=1,
    number_episodes=100,
    use_gui=False,
    csv_sufix="_v3",
    do_save_uncert=False,
):
    sufix = "_U" if use_safe_action else "_NU"
    case = "rerun_test_scenarios" + sufix + csv_sufix
    if do_save_metrics or do_save_uncert:
        with open(filepath + case + ".csv", "w+"):
            pass

    env = Highway(
        sim_params=ps.sim_params,
        road_params=ps.road_params,
        use_gui=use_gui,
        return_more_info=True,
    )

    env.reset()
    if save_video:
        traci_schema(filepath)
    range_ = thresh_range if use_safe_action else list(range(number_tests))

    for thresh in range_:
        episode_rewards = []
        episode_steps = []
        nb_safe_actions_per_episode = []
        nb_safe_hard_actions_per_episode = []
        collissions = 0
        collision_speeds = []
        mean_speeds = []
        stop_events = []
        fast_events = []
        if use_safe_action:
            change_thresh_fn(thresh)
        for i in range(0, number_episodes):
            # np.random.seed(i)
            obs = env.reset()
            if save_video:
                traci_before(filepath, case, thresh, i)
            done = False
            episode_reward = 0
            step = 0
            nb_safe_actions = 0
            nb_hard_safe_actions = 0
            unc = []
            original_unc = []
            uncertainties = [[] for _ in range(10)]
            thresholds = []
            frozen_steps = 0
            stop_vehicle_step = 0
            fast_vehcile_step = 0
            mean_speed = 0

            ep_stop_events = 0
            ep_fast_events = 0
            stop_event = False
            fast_event = False

            events = []

            while done is False:
                action, action_info = dqn.forward(obs)
                if save_video:
                    traci_each(filepath, case, thresh, i, step)
                    # plot_metrics_hist(action_info, filepath, case, thresh, i, step)
                if use_gui or save_video:
                    env.print_info_in_gui2({
                        # "Reward": rewards,
                        "Action": action,
                        "Speed": env.speeds[0, :][0],
                        "Uncertainty": action_info["coefficient_of_variation"][action],
                        "Threshold": thresh,
                        "Safe Action": action_info["safe_action"],
                        "Hard Safe Action": action_info["hard_safe"],
                        "Stop Event": stop_event,
                        "Fast Event": fast_event,
                    }, {})
                obs, rewards, done, _, more_info = env.step(action, action_info)
                reward_no_col = more_info["reward_no_col"]
                mean_speed += more_info["ego_speed"]
                episode_reward += reward_no_col
                step += 1
                if more_info["ego_collision"]:
                    collissions += 1
                    collision_speeds.append(more_info["ego_speed"])
                if "safe_action" in action_info:
                    nb_safe_actions += action_info["safe_action"]
                    nb_hard_safe_actions += action_info["hard_safe"]
                
                if "coefficient_of_variation" in action_info:
                    unc.append(action_info["coefficient_of_variation"][action])
                    for idx, u in enumerate(action_info["coefficient_of_variation"]):
                        uncertainties[idx].append(u)
                    if "max_q_action" in action_info:
                        original_unc.append(action_info["coefficient_of_variation"][action_info["max_q_action"]])
                
                if "threshold" in action_info:
                    thresholds.append(action_info["threshold"])

                if frozen_steps == 0 and np.random.rand() < EVENT_PROBABILITY:
                    if np.random.rand() < STOPPED_VEHICLE_EVENT_PROB:
                        if stop_vehicle():
                            frozen_steps = FREEZE_STEPS + STOP_VEHICLE_STEPS
                            stop_vehicle_step = STOP_VEHICLE_STEPS + 1
                            ep_stop_events += 1
                            stop_event = True
                            events.append({
                                "type": "stop",
                                "step": step,
                            })
                    else:
                        if fast_vehicle():
                            frozen_steps = FREEZE_STEPS + FAST_VEHCILE_STEPS
                            fast_vehcile_step = FAST_VEHCILE_STEPS + 1
                            ep_fast_events += 1
                            fast_event = True
                            events.append({
                                "type": "fast",
                                "step": step,
                            })
                if 0 < frozen_steps:
                    frozen_steps -= 1
                if 0 < stop_vehicle_step:
                    stop_vehicle_step -= 1
                    if stop_vehicle_step == 0:
                        stop_event = False
                        resume_stop_vehicle()
                if 0 < fast_vehcile_step:
                    fast_vehcile_step -= 1
                    if fast_vehcile_step == 0:
                        fast_event = False
                        resume_fast_vehicle()

                
                
            if save_video:
                fig1, ax1 = plt.subplots(ncols=1, nrows=1)
                fig1.set_figwidth(16)
                fig1.set_figheight(16)

                ax1.plot(original_unc, label='Max Q uncertainty', color="r")
                ax1.plot(unc, label='Policy uncertainty', color="b")
                # ax1.axhline(y=thresh, color='m', linestyle="--", label='Threshold')
                ax1.plot(thresholds, color='m', linestyle="--", label='Threshold')
                ax1.set_ylabel("Uncertainty", fontsize=16)
                ax1.set_xlabel("Step", fontsize=16)
                ax1.set_xlim(left=0)

                s_e = 0
                f_e = 0
                for e in events:
                    if e["type"] == "stop":
                        if s_e == 0:
                            ax1.axvspan(e["step"], e["step"] + STOP_VEHICLE_STEPS, alpha=0.1, color='c', label='Stop Event')
                        else:
                            ax1.axvspan(e["step"], e["step"] + STOP_VEHICLE_STEPS, alpha=0.1, color='c')
                        s_e += 1
                    elif e["type"] == "fast":
                        if f_e == 0:
                            ax1.axvspan(e["step"], e["step"] + FAST_VEHCILE_STEPS, alpha=0.1, color='g', label="Fast Event")
                        else:
                            ax1.axvspan(e["step"], e["step"] + FAST_VEHCILE_STEPS, alpha=0.1, color='g')
                        f_e += 1
                ax1.legend()

                name = get_name(filepath)
                fig1.savefig(f"../videos/{name}/{case}/{thresh}_{i}/uncertainties.png")
                plt.close()


                fig1, ax1 = plt.subplots(ncols=1, nrows=1)
                fig1.set_figwidth(16)
                fig1.set_figheight(16)
                for idx, u in enumerate(uncertainties):
                    ax1.plot(u, label=ps.sim_params["action_interp"][idx])
                ax1.legend()
                fig1.savefig(f"../videos/{name}/{case}/{thresh}_{i}/uncertainties_plot.png")
                plt.close()

            if do_save_uncert:
                save_uncert(case, filepath, thresh, i, unc)
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            nb_safe_actions_per_episode.append(nb_safe_actions)
            nb_safe_hard_actions_per_episode.append(nb_hard_safe_actions)
            mean_speeds.append(mean_speed)
            stop_events.append(ep_stop_events)
            fast_events.append(ep_fast_events)
            ep_log(i, step, episode_reward, nb_safe_actions, nb_hard_safe_actions)
        thresh_log(
            thresh,
            episode_rewards,
            episode_steps,
            collissions,
            nb_safe_actions_per_episode,
            nb_safe_hard_actions_per_episode,
        )
        if do_save_metrics:
            save_metrics(
                case,
                filepath,
                thresh,
                episode_rewards,
                episode_steps,
                collissions,
                nb_safe_actions_per_episode,
                nb_safe_hard_actions_per_episode,
                collision_speeds,
                stop_events=stop_events,
                fast_events=fast_events,
            )
    env.close()
