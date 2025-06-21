from envs.env import BasicMultiEnv
from gym.spaces import Box, Discrete, Tuple, Dict
from ray.rllib.evaluation.rollout_worker import get_global_worker
import math
import numpy as np
import cv2


class Vehicle(object):

    def __init__(self):
        self.depart_time = None
        self.first_stop_time = -1
        self.inc = True  # whether on incoming lanes


def rotate_90_degrees(matrix):
    matrix = list(zip(*matrix))
    matrix = [list(row)[::-1] for row in matrix]
    return matrix


def rotate_180_degrees(matrix):
    matrix = [row[::-1] for row in matrix]
    matrix = matrix[::-1]
    return matrix


def rotate_270_degrees(matrix):
    matrix = list(zip(*matrix))
    matrix = matrix[::-1]
    return matrix


class IntelliLight(BasicMultiEnv):

    def __init__(self, scenario, sumo_config, control_config, train_config):
        super().__init__(scenario, sumo_config, control_config, train_config)
        # process net file to get network information
        self.mapping_inc, self.num_int_lane_max, self.mapping_out, _, _, self.out_all = self.scenario.node_mapping()
        self.max_length = self.scenario.max_length_sumolib()
        self.states_tl = self.scenario.get_phases_all_tls()
        # info in the episode
        self.vehicle_info = {}
        self.veh_out_lane = {lane: [] for lane in self.out_all}
        # save info
        save_column_list = ['time', 'tl_id', 'reward',
                            # reward component
                            'sum_queue_length', 'sum_waiting_time', 'sum_delay',
                            'num_veh_passing', 'duration_passing', 'switch',
                            # not used in reward
                            'emergency', 'duration_current',
                            'action',
                            # state component
                            'queue_length', 'waiting_time', 'num_veh', 'now_phase_index']
        self.save_experience = {column: [] for column in save_column_list}

    @property
    def action_space(self):
        return Discrete(2)

    @property
    def observation_space(self):
        max_possible = self.max_length / 4  # grid length
        return Tuple([
            Box(low=0., high=max_possible, shape=(self.num_int_lane_max,)),  # queueLen
            Box(low=0., high=max_possible, shape=(self.num_int_lane_max,)),  # numVeh
            Box(low=0., high=50, shape=(self.num_int_lane_max,)),  # waitTime
            Box(low=0., high=1, shape=(45, 45)),  # mapCar
            Box(low=0., high=11, shape=(2,))  # phase: now + next
        ])

    def _get_state(self):
        """
        - queue length: inc lane num
        - num of vehicles: inc lane num
        - waiting time: inc lane num
        - map feature: 150, 150, 1 grid number
        - current phase: 1
        - next phase: 1
        """
        self.update_vehicle()
        obs = {}
        for tl_id, lanes in self.mapping_inc.items():
            # veh info
            queueLen = [self.sumo.lane.getLastStepHaltingNumber(lane) for lane in lanes]
            numVeh = [self.sumo.lane.getLastStepVehicleNumber(lane) for lane in lanes]
            waitTime = [self.sumo.lane.getWaitingTime(lane) / 60 for lane in lanes]  # in min
            mapCar = self.get_vehicle_map(tl_id)
            # signal info
            now_phase = self.sumo.trafficlight.getRedYellowGreenState(tl_id)
            phases = self.states_tl[tl_id]
            now_phase_index = phases.index(now_phase)
            # next_phase_index = (now_phase_index // 2 * 2 + 2) % len(phases)  # next green phase, for 4 phases, no red
            next_phase_index = 3 * ((now_phase_index // 3 + 1) % 4)  # next green phase, for 12 phases, l+s, red

            obs.update({tl_id: (
                np.array(queueLen),
                np.array(numVeh),
                np.array(waitTime),
                np.array(mapCar),
                np.array([now_phase_index, next_phase_index])
            )})
            self.save_experience['queue_length'].append(str(queueLen))
            self.save_experience['num_veh'].append(str(numVeh))
            self.save_experience['waiting_time'].append(str(waitTime))
            self.save_experience['now_phase_index'].append(now_phase_index)
        return obs

    def _compute_reward(self, action):
        reward = {}
        for tl in self.states_tl.keys():
            # get info on incoming lanes
            inc_lanes = self.mapping_inc[tl]
            queueLen = self.get_junc_queue_length(inc_lanes)  # w1=-0.25
            waitTime = self.get_junc_waiting_time(inc_lanes)  # w2=-0.25
            delay = self.get_junc_delay(inc_lanes)  # w4=-0.25
            # get info for vehicles passing the intersection
            numVehPassing, durationPassing = self.get_junc_veh_passing(tl)  # w5, w6 = 1
            # get info of signal
            assert '1/1' in action.keys(), print('action ERROR', self.sumo.simulation.getTime(), action)
            switch = action[tl]  # w3=-5
            # get info for incoming vehicles
            inc_vehs = self.get_junc_inc_vehs(inc_lanes)
            emergency = self.get_junc_emergency_decel(inc_vehs)
            duration = self.get_junc_travel_duration(inc_vehs)

            # compute reward
            reward_tl = -0.25 * queueLen - 0.25 * waitTime - 5 * switch - 0.25 * delay + numVehPassing + durationPassing
            reward.update({tl: reward_tl})

            # save info
            self.save_experience['time'].append(self.sumo.simulation.getTime())
            self.save_experience['tl_id'].append(tl)
            self.save_experience['reward'].append(reward_tl)
            self.save_experience['sum_queue_length'].append(queueLen)
            self.save_experience['sum_waiting_time'].append(waitTime)
            self.save_experience['sum_delay'].append(delay)
            self.save_experience['num_veh_passing'].append(numVehPassing)
            self.save_experience['duration_passing'].append(durationPassing)
            self.save_experience['switch'].append(switch)
            self.save_experience['emergency'].append(emergency)
            self.save_experience['duration_current'].append(duration)

        return reward

    def _apply_actions(self, actions):
        for agent_id, action in actions.items():
            switch = action > 0  # binary switch
            states = self.states_tl[agent_id]
            now_state = self.sumo.trafficlight.getRedYellowGreenState(agent_id)
            state_index = states.index(now_state)
            if switch and 'G' in now_state:
                self.sumo.trafficlight.setPhase(agent_id, state_index + 1)
            if not switch and 'G' in now_state:
                self.sumo.trafficlight.setPhase(agent_id, state_index)
            self.save_experience['action'].append(1 if switch else 0)

    def reset(self, seed=None, options=None):
        """ Append some specific variables in this env to reset between episodes

        Returns
        -------
        observation: defined in BasicEnv
        """
        obs = super().reset()
        save_column_list = self.save_experience.keys()
        self.save_experience = {column: [] for column in save_column_list}
        self.vehicle_info = {}
        self.veh_out_lane = {lane: [] for lane in self.out_all}
        return obs

    # ---- Specific functions used in this env for traffic control ----

    def get_vehicle_map(self, tl_id, grid_width=4, junction_area=180):
        num_grids = int(junction_area / grid_width)
        mapCar = np.zeros((num_grids, num_grids))

        for veh in self.sumo.vehicle.getIDList():
            veh_position = self.sumo.vehicle.getPosition(veh)  # (double, double) m
            # transform vehicle position to 2D matrix as image representation
            #  ^ veh_position(y)
            #  |
            #  |_____> veh_position(x)
            transformX = math.floor(veh_position[0] / grid_width)
            transformY = math.floor((junction_area - veh_position[1]) / grid_width)
            transformX = min(num_grids - 1, transformX)
            transformY = min(num_grids - 1, transformY)
            mapCar[transformY, transformX] = 1

        return mapCar

    # # reward

    def update_vehicle(self):
        now_veh_list = self.sumo.vehicle.getIDList()

        # remove arrived vehicles
        for arrived_veh in set(self.vehicle_info.keys()) - set(now_veh_list):
            del self.vehicle_info[arrived_veh]

        for veh in now_veh_list:
            # add new vehicles
            if veh not in self.vehicle_info.keys():
                self.vehicle_info.update({veh: Vehicle()})
                self.vehicle_info[veh].depart_time = self.sumo.simulation.getTime()
            # check stop at junction
            speed = self.sumo.vehicle.getSpeed(veh)
            if speed < 0.1 and self.vehicle_info[veh].first_stop_time == -1:
                self.vehicle_info[veh].first_stop_time = self.sumo.simulation.getTime()

    def get_junc_queue_length(self, inc_lanes):
        sum_queue_length = 0
        for lane in inc_lanes:
            sum_queue_length += self.sumo.lane.getLastStepHaltingNumber(lane)
        return sum_queue_length

    def get_junc_waiting_time(self, inc_lanes):
        sum_waiting_time = 0
        for lane in inc_lanes:
            sum_waiting_time += self.sumo.lane.getWaitingTime(lane) / 60  # min
        return sum_waiting_time

    def get_junc_delay(self, inc_lanes):
        sum_speedDiff_percent = 0
        for lane in inc_lanes:
            sum_speedDiff_percent += 1 - self.sumo.lane.getLastStepMeanSpeed(lane) / self.sumo.lane.getMaxSpeed(lane)
        return sum_speedDiff_percent

    def get_junc_inc_vehs(self, inc_lanes):
        inc_vehs = []
        for lane in inc_lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            inc_vehs.extend(veh_list)
        return inc_vehs

    def get_junc_emergency_decel(self, inc_vehs):
        emergency_count = 0
        for veh in inc_vehs:
            accel = self.sumo.vehicle.getAcceleration(veh)
            emergency_count += 1 if accel < -4.5 else 0
        return emergency_count / len(inc_vehs) if inc_vehs else 0

    def get_junc_travel_duration(self, inc_vehs):
        sum_travel_duration = 0
        for veh in inc_vehs:
            sum_travel_duration += (self.sumo.simulation.getTime() - self.vehicle_info[veh].depart_time) / 60
        return sum_travel_duration

    def get_junc_veh_passing(self, tl):
        VehPassing = []
        sum_travel_duration_passing = 0
        out_lanes = self.mapping_out[tl]
        for lane in out_lanes:
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            for veh in vehs:
                if veh not in self.veh_out_lane[lane]:
                    VehPassing.append(veh)
                    if self.vehicle_info[veh].first_stop_time != -1:  # no stop at intersection, not counted
                        sum_travel_duration_passing += (self.sumo.simulation.getTime() -
                                                        self.vehicle_info[veh].first_stop_time) / 60
            self.veh_out_lane[lane] = vehs
        return len(VehPassing), sum_travel_duration_passing


class ImageFreeTL(IntelliLight):

    def __init__(self, scenario, sumo_config, control_config, train_config):
        super().__init__(scenario, sumo_config, control_config, train_config)
        self.green_lanes_per_phase_all_tls = self.scenario.green_lanes_per_phase_all_tls()

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(self.num_int_lane_max+1, ))

    def _get_state(self):
        obs = {}
        self.update_vehicle()

        for tl_id, lanes in self.mapping_inc.items():
            obs.update({tl_id: []})
            # signal info
            now_phase = self.sumo.trafficlight.getRedYellowGreenState(tl_id)
            phases = self.states_tl[tl_id]
            now_phase_index = phases.index(now_phase) / len(phases)
            obs[tl_id].append(now_phase_index)
            for lane in lanes:
                veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
                if veh_list:
                    last_veh = veh_list[-1]
                    veh_list_inverse = veh_list[::-1]
                    dist0_to_junc = (self.sumo.lane.getLength(lane) -
                                     self.sumo.vehicle.getLanePosition(veh_list_inverse[0]))
                    gap = dist0_to_junc  # 90: to skip the first vehicle # compare
                    queue = dist0_to_junc  # save
                    if len(veh_list) > 1:
                        for i in range(1, len(veh_list)):
                            headway = self.sumo.vehicle.getLeader(veh_list_inverse[i])[1]
                            if headway <= gap:
                                queue = (self.sumo.lane.getLength(lane) -
                                         self.sumo.vehicle.getLanePosition(veh_list_inverse[i]))
                                gap = headway
                                last_veh = veh_list_inverse[i]
                            else:
                                break
                    obs[tl_id].append(round((queue + 5)/82.8, 3))  # 5: veh length
                    self.sumo.vehicle.highlight(last_veh, alphaMax=255, duration=1)
                else:
                    obs[tl_id].append(1)

        return obs


class ImageTL3D(IntelliLight):
    # must use sumo-gui

    def __init__(self, scenario, sumo_config, control_config, train_config):
        super().__init__(scenario, sumo_config, control_config, train_config)
        self.green_lanes_per_phase_all_tls = self.scenario.green_lanes_per_phase_all_tls()

    @property
    def observation_space(self):
        return Dict({
            "image": Box(low=0, high=255, shape=(112, 128, 3)),
            "signal": Box(low=0., high=11, shape=(1,))
        })

    def _get_state(self):
        obs = {}
        self.update_vehicle()

        # Get the global RolloutWorker instance
        rollout_worker = get_global_worker()
        # Get the RolloutWorker ID (which is the worker_index)
        try:
            worker_id = rollout_worker.worker_index
        except AttributeError:
            worker_id = 1

        # image preprocessing
        # read image
        image = cv2.imread(f'../training_img/{worker_id}_{self.step_count_in_episode}.jpg')
        assert image is not None, f"cannot read image successfully-{worker_id}_{self.step_count_in_episode}.jpg"
        if image.shape != (765, 1910, 3):
            print(f'REPLACED BY 0： image.shape={image.shape}')
            image = cv2.imread(f'/home/gjy/coach/training_img/0.png')
        # cv2.imshow('Original', image)
        # cv2.waitKey(0)

        # remove the scale in the lower left corner
        image = image[:, 150:, :]
        # crop the image
        image = image[23:742, 445:1164, :]  # dim
        # each inc+out road from camera
        road_N = image[0:303, 317:400, :]
        road_E = image[318:401, 416:, :]
        road_E_rotated = cv2.flip(cv2.transpose(road_E), 0)
        road_S = image[416:, 318:401, :]
        road_S_rotated = cv2.flip(road_S, -1)
        road_W = image[318:401, 0:303, :]
        road_W_rotated = cv2.flip(cv2.transpose(road_W), 1)
        images = [road_N, road_E_rotated, road_S_rotated, road_W_rotated]
        merge_image = np.hstack(images)
        # cv2.imshow('Final', merge_image)
        # cv2.waitKey(0)
        image = cv2.resize(merge_image, (128, 112), interpolation=cv2.INTER_AREA)
        # assert image.shape == (802, 802), print(f'Obs ERROR -{worker_id}_{self.step_count_in_episode}.jpg')
        # cv2.imshow('Resize', image)
        # cv2.waitKey(0)
        cv2.imwrite(f'../training_img/{worker_id}_{self.step_count_in_episode}state.jpg', image)

        # check range
        # max_value = image[0][0][0]
        # print(image.shape)
        # for i in range(128):
        #     for j in range(64):
        #         for k in range(3):
        #             if image[i][j][k] > max_value:
        #                 max_value = image[i][j][k]
        # print(f'max_value: {max_value}')

        # save experience
        for tl_id, lanes in self.mapping_inc.items():
            # veh info
            queueLen = [self.sumo.lane.getLastStepHaltingNumber(lane) for lane in lanes]
            numVeh = [self.sumo.lane.getLastStepVehicleNumber(lane) for lane in lanes]
            waitTime = [self.sumo.lane.getWaitingTime(lane) / 60 for lane in lanes]  # in min
            # signal info
            now_phase = self.sumo.trafficlight.getRedYellowGreenState(tl_id)
            phases = self.states_tl[tl_id]
            now_phase_index = phases.index(now_phase)
            self.save_experience['queue_length'].append(str(queueLen))
            self.save_experience['num_veh'].append(str(numVeh))
            self.save_experience['waiting_time'].append(str(waitTime))
            self.save_experience['now_phase_index'].append(now_phase_index)

            obs.update({tl_id: {"image": image,
                                "signal": np.array([now_phase_index])}})

        return obs




