"""
This script is used to evaluate the SHAP values of the image-based model.
The SHAP values are calculated using the Deep SHAP algorithm, which is a unified approach to explain both
individual predictions and model behavior.
The script is based on the tutorial provided by the SHAP library: https://github.com/slundberg/shap/blob/master/notebooks/image_examples/image_classification/
The script takes the following steps:
<similar to train.py>
1. process SUMO scenario files to get info, required in make env
2. register env and make env in OpenAIgym, then register_env in Ray
3. set DRL algorithm/model
4. multiagent setting
5. assign termination conditions, terminate when achieving one of them
<start XAI evaluation>
6. evaluate the SHAP values of the image-based model
To run the script, use the following command:
python evaluate_SHAP.py --exp_config tl_config.ini
The script will output the SHAP values of the image-based model.
Note: The script is based on the assumption that the image-based model is trained and saved in the ray_results directory.
If the model is not saved, the script will not work.
The SHAP values are calculated for each pixel in the input image, and the results can be visualised using a saliency map.

"""

import os.path
import cv2
import numpy as np
import pandas as pd
import ray
import argparse
import configparser
import sys
import logging

from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.registry import get_policy_class

from scenario.scen_retrieve import SumoScenario
from envs.env_register import register_env_gym
from policies import PolicyConfig

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf

tf.compat.v1.enable_eager_execution()


# ----- Customised functions (multiagent) -----

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # to customise different agent type
    return "policy_0"


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a DRL training for traffic control.",
        epilog="python evaluate_SHAP.py EXP_CONFIG")

    # ----required input parameters----
    parser.add_argument(
        '--exp_config', type=str, default='xTLimage.ini',
        help='Name of the experiment configuration file, as located in exp_configs.')

    # ----optional input parameters----
    parser.add_argument(
        '--log_level', type=str, default='ERROR',
        help='Level setting for logging to track running status.'
    )

    return parser.parse_known_args(args)[0]


def action_dict(action):
    return {'1/1': action}


STATE_OUT = None
# SIGNAL_NOW = 4
FEATURE_NOW = np.zeros(38,)
PHASE_MAPPING = {
    "gGGrgrrrgGGrgrrr": "NST",
    "gyyrgrrrgyyrgrrr": "NSTy",
    "grrrgrrrgrrrgrrr": "Red",
    "grrGgrrrgrrGgrrr": "NSL",
    "grrygrrrgrrygrrr": "NSLy",
    "grrrgGGrgrrrgGGr": "WET",
    "grrrgyyrgrrrgyyr": "WETy",
    "grrrgrrGgrrrgrrG": "WEL",
    "grrrgrrygrrrgrry": "WELy"
}


def main(args):
    args = parse_args(args)
    logging.basicConfig(level=args.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"DRL training with the following CLI args: {args}")
    ray.init()

    # import experiment configuration
    config_file = args.exp_config
    config = configparser.ConfigParser()
    config.read(os.path.join('./exp_configs', config_file))
    if not config:
        logger.error(f"Unable to find the experiment configuration {config_file} in exp_configs")
    config.set('TRAIN_CONFIG', 'log_level', args.log_level)

    # 1. process SUMO scenario files to get info, required in make env
    scenario = SumoScenario(config['SCEN_CONFIG'])
    logger.info(f"The scenario cfg file is {scenario.cfg_file_path}.")

    # 2. register env and make env in OpenAIgym, then register_env in Ray
    this_env = __import__("envs", fromlist=[config.get('TRAIN_CONFIG', 'env')])
    if hasattr(this_env, config.get('TRAIN_CONFIG', 'env')):
        this_env = getattr(this_env, config.get('TRAIN_CONFIG', 'env'))
    this_env_register, env_name = register_env_gym(this_env, scenario, config['SUMO_CONFIG'], config['CONTROL_CONFIG'],
                                                   config['TRAIN_CONFIG'])
    register_env(env_name, this_env_register)
    this_env = this_env(scenario, config['SUMO_CONFIG'], config['CONTROL_CONFIG'], config['TRAIN_CONFIG'])

    # 3. set DRL algorithm/model
    configs_to_ray = PolicyConfig(env_name, config['ALG_CONFIG'], config['TRAIN_CONFIG'], config['MODEL_CONFIG']).policy

    # 4. multiagent setting
    policies = {}
    act_space_dict = {}
    obs_space_dict = {}
    for section in config.sections():
        if 'policySpec' in section:
            policy_class = config.get(section, 'policy_class', fallback=None)
            if policy_class:
                policy_class = get_policy_class(policy_class)
            obs_space = getattr(this_env, config.get(section, 'observation_space'), None)
            act_space = getattr(this_env, config.get(section, 'action_space'), None)
            num_agents = config.getint(section, 'num_agents', fallback=1)
            for i in range(num_agents):
                num_policies = len(policies.keys())
                policies.update({'policy_' + str(num_policies): PolicySpec(policy_class, obs_space, act_space,
                                                                           {'agent_id': num_policies})})
                act_space_dict.update({str(num_policies): act_space})
                obs_space_dict.update({str(num_policies): obs_space})
    for _, spec in policies.items():
        spec.config.update({"act_space_dict": act_space_dict})
        spec.config.update({"obs_space_dict": obs_space_dict})

    if policies:
        configs_to_ray.update({"multiagent": {"policies": policies,
                                              "policy_mapping_fn": policy_mapping_fn,
                                              "policies_to_train": list(policies.keys())}})
    configs_to_ray.update({'disable_env_checking': True})  # to avoid checking non-override default obs_space...
    # configs_to_ray.update({'eager_tracing': False})  # For debugging purposes

    # 5. assign termination conditions, terminate when achieving one of them
    stop_conditions = {}
    for k, v in config['STOP_CONFIG'].items():
        stop_conditions.update({k: int(v)})

    # 6. Evaluation
    from ray.rllib.policy.policy import Policy

    policy = Policy.from_checkpoint(checkpoint='../ray_results/ImageTL3D_3turnRed/'
                                               'PPO_2024-07-07_14-01-20/'
                                               'PPO_ImageTL3D-v0_01429_00000_0_2024-07-07_14-01-20/'
                                               'checkpoint_000014/policies/policy_0')

    # simulate simple env
    obs = this_env.reset()

    # RNN requried
    initial_state = [np.zeros(256, np.float32),
                     np.zeros(256, np.float32)]

    action, state_out, _ = policy.compute_single_action(obs=obs[0]['1/1'], state=initial_state)
    save_actions = []
    save_rewards = []
    save_actions_prob = []
    # warmup scenario
    for _ in range(5):
        obs, rewards, _, _, _ = this_env.step(action_dict(action))
        save_actions.append(action)
        save_rewards.append(rewards['1/1'])
        action, state_out, action_info = policy.compute_single_action(obs=obs['1/1'], state=state_out)
        action_prob = action_info["action_prob"]
        if action == 1:
            save_actions_prob.append(action_prob)

    # --- Image ---
    def predict(obs):
        output_list = []
        for n in range(obs.shape[0]):  # shape[0] <= max_batch
            each_obs = obs[n, :, :, :]
            # cv2.imshow('Masked', each_obs)
            # cv2.waitKey(0)
            observation = {'image': each_obs, 'feature': FEATURE_NOW}
            action_dist_inputs = policy.compute_single_action(obs=observation, state=STATE_OUT)[2]["action_dist_inputs"]
            output_list.append(action_dist_inputs)
        return np.array(output_list)

    # XAI - steps
    for i in range(650):
        obs, rewards, _, _, _ = this_env.step(action_dict(action))
        save_actions.append(action)
        save_rewards.append(rewards['1/1'])
        action, state_out, action_info = policy.compute_single_action(obs=obs['1/1'], state=state_out)
        action_prob = action_info["action_prob"]
        if action == 1:
            save_actions_prob.append(action_prob)
        STATE_OUT = state_out
        FEATURE_NOW = obs['1/1']['feature']
        # SIGNAL_NOW = obs['1/1']['signal']
        # get mask
        image_obs = obs['1/1']['image']
        input_img = np.reshape(image_obs, (1, 64, 256, 3))  # only one image to process
        masker_blur = shap.maskers.Image("blur(4, 4)", image_obs.shape)
        explainer = shap.Explainer(predict, masker_blur, output_names=['0', '1'])
        shap_values = explainer(input_img, max_evals=5000, batch_size=500, outputs=[action])
        # shap.Explanation.argsort.flip[:2]  [action]
        # outputs for output_names index
        # shap.image_plot(shap_values=shap_values.values,
        #                 pixel_values=shap_values.data,
        #                 labels=shap_values.output_names)

        now_phase = this_env.sumo.trafficlight.getRedYellowGreenState('1/1')

        shap.image_plot(shap_values, image_id=str(i + 1),
                        action_taken=f'{action}({str(round(action_prob, 3))})',
                        phase=PHASE_MAPPING[now_phase],
                        # hspace='auto',
                        )

    # run simulation til the end
    for _ in range(345):
        obs, rewards, _, _, _ = this_env.step(action_dict(action))
        save_actions.append(action)
        save_rewards.append(rewards['1/1'])
        action, state_out, action_info = policy.compute_single_action(obs=obs['1/1'], state=state_out)
        action_prob = action_info["action_prob"]
        if action == 1:
            save_actions_prob.append(action_prob)

    print(f'episode reward: {sum(save_rewards)}')
    print(f'avg action: {np.mean(save_actions)}')

    dataframe = pd.DataFrame({"prob": save_actions_prob})
    print(dataframe.describe())

    # --- save action_prob ---
    # data = {'timestep': [i for i in range(385)],
    #         'action': save_actions,
    #         'action_prob': save_actions_prob}
    # import csv
    # with open('action_probDist.csv', "w") as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(data.keys())
    #     writer.writerows(zip(*data.values()))

    ray.shutdown()
    create_video()


def create_video():
    image_folder = '../video'
    video_name = 'SHAP_650.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(filename=video_name, fourcc=0x7634706d, fps=2, frameSize=(width, height))
    # img_prefix = images[0].split('_')[0]  # img_prefix+'_'+
    img_suffix = images[0].split('.')[1]
    for i in range(len(images)):
        video.write(cv2.imread(os.path.join(image_folder, str(i + 1) + '.' + img_suffix)))
    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main(sys.argv[1:])
    # create_video()
