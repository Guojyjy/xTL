import csv
import logging
import os
import sys
import time
from typing import Tuple

import gym
from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.evaluation.rollout_worker import get_global_worker

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import numpy as np
from ray.rllib.env import MultiAgentEnv

from utils.file_processing import ensure_dir

RETRIES_ON_ERROR = 10  # Number of retries on restarting SUMO before giving up
LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ
logger = logging.getLogger(__name__)


class BasicEnv(gym.Env):
    """

    """

    def __init__(self, scenario, sumo_config, control_config, train_config):
        """

        Parameters
        ----------
        scenario
        sumo_config
        control_config
        train_config
        """
        logger.setLevel(train_config.get('log_level'))
        self.name = train_config.get('exp_name')

        self.scenario = scenario
        self.sumo = None  # sumo simulation API
        self.num_episode = 0
        self.step_count_in_episode = 0

        self.sim_step = sumo_config.getfloat('sim_step', fallback=1)
        self.sumo_gui = sumo_config.getboolean('sumo-gui', fallback=False)
        self.seed = sumo_config.get('seed', fallback='random')
        self.output_path = sumo_config.get('output_path', fallback=None)
        self.num_output = sumo_config.getint('num_output', fallback=10)
        self.warmup_steps = train_config.getint('warmup_steps', fallback=0)
        self.horizon = train_config.getint('horizon')

        self.save_experience = {}

    @property
    def action_space(self):
        """


        Returns
        -------
        gym Box or Tuple type
        """
        pass

    @property
    def observation_space(self):
        """

        Returns
        -------
        gym Box or Tuple type
        """
        pass

    def _get_state(self):
        """ (Required)

        Returns
        -------
        states: agent's observation of the current environment
        """
        pass

    def _compute_reward(self):
        """ (Required)

        Returns
        -------
        reward: float or a list of float
        """
        pass

    def _compute_dones(self):
        """ (Required for vehicle control)

        Returns
        -------
        done: bool or a dict of each agent to determine whether the episode has ended
        """
        return self.step_count_in_episode >= self.horizon + self.warmup_steps

    def _get_info(self):
        """ (Optional)

        Returns
        -------
        info: contains auxiliary diagnostic information (helpful for debugging, logging, and sometimes learning)
        """
        pass

    def _apply_actions(self, action):
        """ convert action assigned to agent into traffic control (Required) """
        pass

    def _additional_command(self):
        """Additional commands that may be performed by the step method (Optional)
        """
        pass

    def _start_sumo(self):
        if self.sumo:
            traci.close()
            self.sumo = None

        for _ in range(RETRIES_ON_ERROR):
            try:
                sumo_binary = "sumo-gui" if self.sumo_gui else "sumo"
                sumo_cmd = [sumo_binary, "-c", self.scenario.cfg_file_path,
                            "--step-length", str(self.sim_step),
                            "--no-warnings",  # teleport, or collision
                            "--scale", "1",
                            "--tripinfo-output.write-unfinished",
                            "-S",  # Start the simulation after loading
                            "-Q",  # Quits the GUI when the simulation stops
                            ]

                file_index = self.num_episode % self.num_output
                if file_index == self.num_output - 1:  # to reduce stored output files, e.g., each 10 iterations
                    print('save output at', self.num_episode, file_index)
                    if self.output_path is not None:
                        ensure_dir(self.output_path)

                        output_filetypes = ["emission-output", "tripinfo-output", "queue-output"]
                        this_time = str(time.time())
                        for each in output_filetypes:
                            file_name = os.path.join(self.output_path,
                                                     f"{self.name}-{this_time}-{each.split('-')[0]}.xml")
                            sumo_cmd.extend([f"--{each}", file_name])

                        # obtain safety info
                        file_name = os.path.join(os.getcwd().split('envs')[0], self.output_path,
                                                 f"{self.name}-{this_time}-ssm.xml")

                        sumo_cmd.extend(["--device.ssm.deterministic", "true"])
                        sumo_cmd.extend(["--device.ssm.measures", "TTC DRAC PET BR SGAP TGAP"])
                        sumo_cmd.extend(["--device.ssm.thresholds", "3.0 3.0 2.0 0.0 0.2 0.5"])
                        sumo_cmd.extend(["--device.ssm.range", "50.0"])
                        sumo_cmd.extend(["--device.ssm.extratime", "5.0"])
                        sumo_cmd.extend(["--device.ssm.trajectories", "true"])
                        sumo_cmd.extend(["--device.ssm.geo", "false"])
                        sumo_cmd.extend(["--device.ssm.file", file_name])


                if self.seed == "random":
                    sumo_cmd.append("--random")
                else:
                    sumo_cmd.extend(["--seed", self.seed])

                logger.info(f"Connect to SUMO: {sumo_cmd}")

                # wait a small period of time for the subprocess to activate before trying to connect with Traci
                time.sleep(1)

                if LIBSUMO:
                    traci.start(sumo_cmd, numRetries=100)
                else:
                    traci.start(sumo_cmd, numRetries=100, label=str(time.time()))
                self.num_episode += 1
                self.sumo = traci
                self.scenario.sumo = traci

                break
            except Exception as e:
                logger.exception(f"Error during starting a SUMO instance: {e}")

    def reset(self, **kwargs):
        """Reset the environment and returns observations from ready agents
        (Optional to append other variables reset between episodes)

        Preform in between simulation episodes.
        It reset the state of the environment.

        Parameters
        ----------
        **kwargs

        Returns
        -------
        observation: dict of array_like
            the initial observation of the environment
            And the initial reward is assumed to be zero

        """
        self.step_count_in_episode = 0

        self._start_sumo()
        self.sumo.simulationStep()

        obs = self._get_state()

        # perform (optional) warm-up steps before training
        for _ in range(self.warmup_steps):
            obs, _, _, _ = self.step(action=None)

        return obs

    def step(self, action):
        """Advance the environment by one step

        Parameters
        ----------
        action: an action provided by the agent or dict of array_like
            actions provided by the  agents

        Returns
        -------
        observation: object for single-agent or dict for multiagent
            agent's observation of the current environment, i.e., state
        reward: float or dict
            amount of reward associated with the previous state/action pair
        done: bool or dict of each agent to determine whether the episode has ended
            done value (bool) for each agent
            "__all__" indicates whether the episode has ended in dict
        info: dict,
            contains other diagnostic information from the previous action, optional values for each agent
        """
        self.step_count_in_episode += 1
        self._apply_actions(action)
        self._additional_command()
        self.sumo.simulationStep()

        obs = self._get_state()
        reward = self._compute_reward()
        dones = self._compute_dones()
        info = self._get_info()

        return obs, reward, dones, info

    def render(self, mode="human"):
        pass

    def close(self):
        if self.sumo:
            self.sumo.close()
            self.sumo = None

    def experience_to_csv(self):
        """
        save info component of state, reward, action for XAI
        """
        this_time = str(time.time())
        file_name = os.path.join(os.getcwd().split('envs')[0], self.output_path,
                                 f"{self.name}-{this_time}-experience.csv")
        try:
            ensure_dir(os.path.join(os.getcwd().split('envs')[0], self.output_path))
        except OSError:
            pass

        with open(file_name, "w") as f:
            print(f'File saved: {file_name}')
            writer = csv.writer(f, delimiter=',')
            writer.writerow(self.save_experience.keys())
            writer.writerows(zip(*self.save_experience.values()))


class BasicMultiEnv(BasicEnv, MultiAgentEnv):

    def reset(self, **kwargs):
        """Reset the environment and returns observations from ready agents
        (Optional to append other variables reset between episodes)

        Preform in between simulation episodes.
        It reset the state of the environment.

        Parameters
        ----------
        **kwargs

        Returns
        -------
        observation: dict of array_like
            the initial observation of the environment
            And the initial reward is assumed to be zero

        """
        if self.save_experience:
            file_index = self.num_episode % self.num_output
            if file_index == self.num_output - 1:
                self.experience_to_csv()
            save_keys = self.save_experience.keys()
            self.save_experience.update({key: [] for key in save_keys})

        self.step_count_in_episode = 0

        self._start_sumo()
        if self.sumo_gui:
            # Get the global RolloutWorker instance
            rollout_worker = get_global_worker()
            # Get the RolloutWorker ID (which is the worker_index)
            try:
                worker_id = rollout_worker.worker_index
            except AttributeError:
                worker_id = 1
            # self.sumo.gui.setZoom(viewID='View #0', zoom=100)
            # self.sumo.gui.setSchema(viewID='View #0', schemeName="real world")
            self.sumo.gui.screenshot(viewID='View #0',
                                     filename=f'../training_img/{worker_id}_0.jpg')
        self.sumo.simulationStep()

        obs = self._get_state()
        info = self._get_info()

        # perform (optional) warm-up steps before training
        for _ in range(self.warmup_steps):
            obs, _, _, info = self.step(action=dict())

        return obs, info

    def step(self, action):
        """Advance the environment by one step

        Parameters
        ----------
        action: an action provided by the agent or dict of array_like
            actions provided by the  agents

        Returns
        -------
        observation: object for single-agent or dict for multiagent
            agent's observation of the current environment, i.e., state
        reward: float or dict
            amount of reward associated with the previous state/action pair
        done: bool or dict of each agent to determine whether the episode has ended
            done value (bool) for each agent
            "__all__" indicates whether the episode has ended in dict
        info: dict,
            contains other diagnostic information from the previous action, optional values for each agent
        """
        self.step_count_in_episode += 1
        self._apply_actions(action)
        self._additional_command()

        if self.sumo_gui:
            # Get the global RolloutWorker instance
            rollout_worker = get_global_worker()
            # Get the RolloutWorker ID (which is the worker_index)
            try:
                worker_id = rollout_worker.worker_index
            except AttributeError:
                worker_id = 1
            # save image
            self.sumo.gui.screenshot(viewID='View #0',
                                     filename=f'../training_img/{worker_id}_{self.step_count_in_episode}.jpg')

        self.sumo.simulationStep()

        obs = self._get_state()
        reward = self._compute_reward(action)  # for IntelliLight, TL switch
        dones = self._compute_dones()
        truncateds = self._compute_truncateds()
        info = self._get_info()

        return obs, reward, dones, truncateds, info

    # require for multi-agent env
    def _compute_dones(self):
        # termination conditions for the environment
        done = {}
        if self.step_count_in_episode >= (self.horizon + self.warmup_steps):
            done['__all__'] = True
        else:
            done['__all__'] = False

        return done

    def _compute_truncateds(self):
        truncateds = {}
        if self.step_count_in_episode >= (self.horizon + self.warmup_steps):
            truncateds['__all__'] = True
        else:
            truncateds['__all__'] = False
        return truncateds

    def _get_info(self):
        # Key set for infos must be a subset of obs
        return {}
