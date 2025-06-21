from ray.rllib.models import ModelCatalog
from policies.models.CNNModel import CNNModelLSTMSignal
import ast


def model_create(model_configs):
    model = {}
    for key in model_configs.keys():
        if key == 'name':
            model.update({'custom_model': model_configs.get(key)})
            ModelCatalog.register_custom_model(
                model_configs.get('name'),
                CNNModelLSTMSignal,
            )
        else:
            # model.update({'conv_filters': [[16, [4, 4], 2], [32, [4, 4], 2], [256, [4, 4], 2]]})
            try:
                model.update({key: ast.literal_eval(model_configs.get(key))})
            except ValueError:
                print(f"model config error: {key}")

    return model


def ppo_config(train_configs, env_name, model_configs):
    model = model_create(model_configs)

    return {
        "env": env_name,
        "log_level": train_configs.get('log_level'),
        "num_workers": train_configs.getint('num_workers'),
        "train_batch_size": train_configs.getint('horizon') * train_configs.getint('num_workers') * 3
        if not train_configs.getint('train_batch_size') else train_configs.getint('train_batch_size'),
        "gamma": 0.999,  # discount rate
        "model": model,
        "use_gae": True,
        "lambda": 0.97,
        "kl_target": 0.02,
        "num_sgd_iter": 10,
        "horizon": train_configs.getint('horizon'),
        "timesteps_per_iteration": train_configs.getint('horizon') * train_configs.getint('num_workers') * 3,
        "no_done_at_end": True,
    }


def dqn_config(train_configs, env_name, model_configs):
    model = model_create(model_configs)

    return {
        "env": env_name,
        "log_level": train_configs.get('log_level'),
        "num_workers": train_configs.getint('num_workers'),
        "train_batch_size": 1240,
        "horizon": train_configs.getint('horizon'),
        "rollout_fragment_length": 385,
        "gamma": 0.95,  # discount rate
        "exploration_fraction": 0.05,
        "timesteps_per_iteration": train_configs.getint('horizon') * train_configs.getint('num_workers')
    }


class PolicyConfig:

    def __init__(self, env_name, alg_configs, train_configs, model_configs=None):
        self.env_name = env_name
        self.name = alg_configs.get('alg_name')
        self.policy = self.find_policy(train_configs, model_configs)

    def find_policy(self, train_configs, model_configs):
        if self.name == 'PPO':
            return ppo_config(train_configs, self.env_name, model_configs)
        elif self.name == 'DQN':
            return dqn_config(train_configs, self.env_name, model_configs)
        return None
