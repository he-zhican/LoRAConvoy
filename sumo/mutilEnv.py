import math
import os
import random
import cv2
import gym
import numpy as np
from gym import spaces
import traci
from sumo.convoyVehicle import ConvoyVehicle
from sumo.road import Road
from sumo.vehicle import Vehicle


class MutilEnv(gym.Env):
    def __init__(self, render_mode=None,sumo_home="", result_folder=""):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.render_mode = render_mode
        self.result_folder = result_folder
        self.result_files = ""
        # os.environ['SUMO_HOME'] = '/usr/share/sumo'
        # Define the SUMO executable path
        self.client = traci
        if render_mode == "human":
            self.sumo_binary = sumo_home+"sumo-gui"  # If you want to use GUI mode, use "sumo-gui", otherwise use "sumo"
        else:
            self.sumo_binary = sumo_home+"sumo"
        # A SUMO configuration file containing map and simulation parameters
        self.sumo_config = f"configs/convoy.sumocfg"
        # self.client.start([self.sumo_binary, "-c", self.sumo_config, "--start"])
        self.client.start([self.sumo_binary, "-c", self.sumo_config])
        self.cv_names = ["veh1", "veh2", "veh3", "veh4", "veh5", "veh6", "veh7", "veh8"]
        # self.cv_names = ["veh1"]
        self.desired_lanes = [1, 0, 1, 0, 1, 0, 1, 0]
        self.aver_speeds = [0] * 8  # The average speed of all convoy vehicles
        self.convoy_vehicles = []  # All convoy vehicles
        # Each time a decision is made by the LLM, the environment executes 10 time steps
        self.decision_frequency = 10
        self.dt = 0.025
        self.video_writer = None
        # The width and height of the video frame
        self.width = 1148
        self.height = 800

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        episode = kwargs["episode"]
        if seed is not None:
            random.seed(seed)
        self.convoy_vehicles.clear()
        self.video_writer = None
        # reload the environment
        self.client.load(
            ["-c", f"configs/convoy.sumocfg", "--start", "--seed", f"{seed}", "--lanechange.duration",
             "1.5"])
        self.client.simulationStep()

        self.result_files = [self.result_folder + f'/speed_{episode}.txt', self.result_folder + f'/position_error_{episode}.txt']

        for file_path in self.result_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Initialize all convoy vehicles
        self.convoy_vehicles = self.get_all_cvs()
        # The perspective is focused on veh5
        if self.render_mode == "human":
            self.client.gui.trackVehicle("View #0", "veh5")
            self.client.gui.setZoom("View #0", 1000)
        # Update the states of the self and environment vehicles
        states = self.get_state()
        return states, {}

    def step(self, actions):
        # Action Decoder
        for i, cv in enumerate(self.convoy_vehicles):
            action = self.get_avail_actions(cv, actions[i])
            if action == 1:  # maintain
                # cv.target_lane = cv.lane
                pass
            elif action == 0:  # turn left
                cv.target_lane = cv.lane + 1
            elif action == 2:  # turn right
                cv.target_lane = cv.lane - 1
            elif action == 3:  # accelerate
                cv.target_speed = cv.last_target_speed + 1.0
            elif action == 4:  # decelerate
                cv.target_speed = cv.last_target_speed - 2.0

        # 生成环境车辆
        self.spawn_environment_vehicles()

        states = []
        rewards = []
        dones = []
        truncateds = []
        for _ in range(self.decision_frequency):
            self.write_data()
            states.clear()
            rewards.clear()
            dones.clear()
            truncateds.clear()
            for i, cv in enumerate(self.convoy_vehicles):
                reward, done, truncated = self.get_reward(cv)
                rewards.append(reward)
                dones.append(done)
                truncateds.append(truncated)
                cv.planning(self.convoy_vehicles)
                self.client.vehicle.moveToXY(cv.id, "", cv.lane, cv.new_x, cv.new_y,
                                             angle=90 - cv.heading * 180 / math.pi, keepRoute=2)
                self.aver_speeds[i] = round((self.aver_speeds[i] + cv.speed) / 2, 2)
            self.client.simulationStep()
            states = self.get_state()
            if True in dones or True in truncateds:
                break

            if self.render_mode == "human":
                self.render()

        return states, rewards, dones, truncateds, {}

    def get_state(self):
        # Update the environment vehicles
        all_evs = self.get_all_evs()
        # Update the convoy vehicles, their neighborhoods and surround_vehicles
        for cv in self.convoy_vehicles:
            cv.update_state(self.client)
            cv.find_surround_evs(all_evs)
            cv.find_neighborhoods(self.convoy_vehicles)

        return []

    def get_reward(self, convoy_vehicle):
        done = False  # Termination due to collision
        truncated = False  # Termination caused by the convoy vehicle driving the entire road
        reward = 0

        for neighbor in convoy_vehicle.neighborhoods:
            if neighbor is not None:
                # collision with neighborhoods
                if Road.check_collision(convoy_vehicle, neighbor):
                    done = True
                    break

        for ev in convoy_vehicle.surround_evs:
            if ev is not None:
                # collision with environment vehicles
                if Road.check_collision(convoy_vehicle, ev):
                    done = True
                    break

        return reward, done, truncated

    # Calculate the formation position error for each convoy vehicle
    def get_position_error(self):
        position_errors = []
        for cv in self.convoy_vehicles:
            position_error = 0
            for i, neighbor in enumerate(cv.neighborhoods):
                if neighbor is not None:
                    dis = abs(Road.relation_distance(cv.x, cv.y, neighbor.x, neighbor.y))
                    if i == 2 or i == 3:
                        position_error += round(abs(dis - cv.safe_distance), 2)
                    else:
                        position_error += round(abs(dis - cv.safe_distance / 2), 2)
            position_errors.append(position_error)
        return position_errors

    def write_data(self):
        with open(self.result_files[0], 'a') as file:
            for cv in self.convoy_vehicles:
                file.write(f'{round(cv.speed, 2)},')
            file.write('\n')
        with open(self.result_files[1], 'a') as file:
            position_errors = self.get_position_error()
            for position_error in position_errors:
                file.write(f'{round(position_error, 2)},')
            file.write('\n')

    def get_avail_actions(self, cv, action):
        if (cv.lane == 2 and action == 0) or (cv.lane == 0 and action == 2):
            action = 1
        elif (cv.speed <= 1 and action == 4) or (cv.speed >= 30 and action == 3):
            action = 1

        return action

    # Get all convoy vehicles
    def get_all_cvs(self):
        all_cvs = []
        for i, cv_id in enumerate(self.cv_names):
            temp_v = ConvoyVehicle(cv_id, self.desired_lanes[i])
            temp_v.update_state(self.client)
            self.aver_speeds[i] = temp_v.speed  # Initialize the average speed of each vehicle
            all_cvs.append(temp_v)

        return all_cvs

    # Get all the environment vehicles in SUMO
    def get_all_evs(self):
        all_evs = []
        vehicles_list = self.client.vehicle.getIDList()
        for c in vehicles_list:
            if c not in self.cv_names:
                temp_v = Vehicle(c)
                temp_v.update_state(self.client)
                all_evs.append(temp_v)
        return all_evs

    def get_all_lead_cvs(self):
        all_lead_cvs = []
        # Get the lead vehicles in the convoy
        for cv in self.convoy_vehicles:
            if cv.find_neighborhoods(self.convoy_vehicles)[2] is None:
                all_lead_cvs.append(cv)
        return all_lead_cvs

    def get_first_convoy_vehicle(self):
        """
        Find the first vehicle in the convoy - the one that is furthest ahead.
        Returns the leading convoy vehicle.
        """
        # Get all lead vehicles in the convoy
        lead_cvs = self.get_all_lead_cvs()

        if not lead_cvs:
            return None

        # If only one lead vehicle, return it directly
        if len(lead_cvs) == 1:
            return lead_cvs[0]

        # Among multiple lead vehicles, find the one that is most forward
        first_cv = lead_cvs[0]
        for i in range(1, len(lead_cvs)):
            # Compare positions using Road.front_or_behind
            # If front_or_behind returns positive value, lead_cvs[i] is ahead of first_cv
            relation = Road.front_or_behind(first_cv.x, first_cv.y, lead_cvs[i].x, lead_cvs[i].y)
            if relation == 1:  # lead_cvs[i] is ahead of first_cv
                first_cv = lead_cvs[i]

        return first_cv

    def spawn_environment_vehicles(self, ev_num=10):
        """
        Spawn environment vehicles ahead of the first convoy vehicle
        """
        # Get the first vehicle in the convoy
        first_cv = self.get_first_convoy_vehicle()

        if first_cv is None:
            return

        sim_time = self.client.simulation.getTime()

        if abs(sim_time % 10 - 0.025) < 1e-10:
            all_evs = self.get_all_evs()
            front_evs_num = Road.vehicles_ahead_num(first_cv, all_evs)
            if front_evs_num < ev_num:
                Road.spawn_vehicles_ahead(self.client, first_cv, sim_time,
                                          ev_num - front_evs_num,
                                          min_gap=60, max_gap=300)

    def render(self):
        screenshot_filename = f"screenshots/screenshot.png"
        # Call the traci.gui.screenshot method to take a screenshot
        self.client.gui.screenshot("View #0", screenshot_filename, width=self.width, height=self.height)
        # Read the picture and write it to the video file
        img = cv2.imread(screenshot_filename)
        if img is not None:
            self.video_writer.write(img)

    # Initialize the video writer
    def initialize_video_writer(self, video_path):
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V encoding
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (self.width, self.height))

    def close_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()

    def get_aver_speeds(self):
        return self.aver_speeds

    def close(self):
        self.client.close()
