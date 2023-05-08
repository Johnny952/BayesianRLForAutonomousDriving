import csv
import numpy as np
import os
import shutil
import sys

sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), 'tools'))
import traci

from base.driving_env import Highway

# test scenarios
NB_VEHICLES = 25 # 25
speed_range = [15, 25] # [15, 35]

# Standstill
stop_position_range = [280, 500]  # 300 = 280, 500
stop_speed_range = [0, 10]  # 0
vehicle_distance_range = [12, 22]  # 18
vehicle_start_pos_range = [0, 36]  # 36
vehicles_speed_range = [10, 20]  # 15

# Fast overtaking
vehicles_speeds_range_fast = [10, 20]  # 15
fast_vehicle_speed_range = [40, 60]  # 50
fast_vehicle_start_position_range = [100, 200]  # 150

def get_name(filepath):
    return filepath.split('/')[-2]

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
):
    with open(filepath + case + '.csv', "a+") as file:
        writer = csv.writer(file)
        mean_speeds = 0 if len(collision_speeds) == 0 else np.mean(collision_speeds)
        writer.writerow(
            [
                thresh,
                np.sum(episode_rewards) / np.sum(episode_steps),
                collissions / len(episode_steps),
                np.sum(nb_safe_actions_per_episode) / np.sum(episode_steps),
                np.sum(nb_safe_hard_actions_per_episode) / np.sum(episode_steps),
                mean_speeds,
                np.sum(episode_steps),
            ]
        )

def save_uncert(
    case,
    filepath,
    thresh,
    episode,
    uncert,
):
    with open(filepath + case + '.csv', "a+") as file:
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

    traci.gui.setZoom("View #0", 10500)
    traci.gui.screenshot("View #0", video_folder + "/init.png")

def traci_each(filepath, case, thresh, episode, step):
    name = get_name(filepath)
    video_folder = f"../videos/{name}/{case}/{thresh}_{episode}/"
    traci.gui.screenshot(
        "View #0", video_folder + str(step) + ".png"
    )


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
    csv_sufix='',
    do_save_uncert=False,
):
    sufix = '_U' if use_safe_action else '_NU'
    case = 'rerun_test_scenarios' + sufix + csv_sufix
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
                    episode_reward += rewards
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
    csv_sufix='',
    do_save_uncert=False,
):
    sufix = '_U' if use_safe_action else '_NU'
    case = 'fast_overtaking' + sufix + csv_sufix
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
                traci.vehicle.moveTo("veh2", "highway_1", s0 - fast_vehicle_start_position)
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
            for _ in range(8):
                try:
                    action, action_info = dqn.forward(observation)
                    observation, reward, done, _, more_info = env.step(action, action_info)
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
    csv_sufix='',
    do_save_uncert=False,
):
    sufix = '_U' if use_safe_action else '_NU'
    case = 'standstill' + sufix + csv_sufix
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
                s0 + np.random.uniform(stop_position_range[0], stop_position_range[1]),
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

            # Run standstill case
            if save_video:
                traci_before(filepath, case, thresh, i)
            nb_safe_actions = 0
            nb_hard_safe_actions = 0

            episode_reward = 0
            unc = []
            step = 0
            for _ in range(30):
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
