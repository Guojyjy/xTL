from typing import Dict, List

import numpy as np
from ray.rllib.models.tf import TFModelV2
from ray.rllib.utils.typing import TensorType

import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer


class CNNModelLSTMSignal(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CNNModelLSTMSignal, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        image_input = tf.keras.layers.Input(shape=(112, 128, 3), name="observation")
        signal_input = tf.keras.layers.Input(shape=(1,), name="signal_input")

        cnn_layer_1 = tf.keras.layers.Conv2D(filters=1,
                                             kernel_size=2,
                                             strides=(2, 2),
                                             activation="relu",
                                             padding="valid",
                                             data_format="channels_last",
                                             name="conv_out_1",
                                             )(image_input)
        cnn_layer_2 = tf.keras.layers.Conv2D(filters=1,
                                             kernel_size=2,
                                             strides=(2, 2),
                                             activation="relu",
                                             padding="valid",
                                             data_format="channels_last",
                                             name="conv_out_2",
                                             )(cnn_layer_1)
        cnn_layer_3 = tf.keras.layers.Conv2D(filters=1,
                                             kernel_size=2,
                                             strides=(2, 2),
                                             activation="relu",
                                             padding="valid",
                                             data_format="channels_last",
                                             name="conv_out_3",
                                             )(cnn_layer_2)
        flatten = tf.keras.layers.Flatten(data_format="channels_last")(
            cnn_layer_3
        )

        # Last hidden layer output (before logits outputs).
        last_layer = tf.concat([flatten, signal_input], axis=1)
        # The action distribution outputs.
        logits_out = None

        last_layer = tf.keras.layers.Dense(32,
                                           name="fc_1",
                                           activation='tanh',
                                           kernel_initializer=normc_initializer(1.0)
                                           )(last_layer)
        last_layer = tf.keras.layers.Dense(32,
                                           name="fc_2",
                                           activation='tanh',
                                           kernel_initializer=normc_initializer(1.0)
                                           )(last_layer)
        last_layer = tf.keras.layers.Dense(32,
                                           name="fc_3",
                                           activation='tanh',
                                           kernel_initializer=normc_initializer(1.0)
                                           )(last_layer)

        if num_outputs:
            logits_out = tf.keras.layers.Dense(num_outputs,
                                               name="fc_out",
                                               activation=None,
                                               kernel_initializer=normc_initializer(0.01)
                                               )(last_layer)
        else:
            # LSTM_wrapper set num_outputs = None
            self.num_outputs = last_layer.shape[1]

        self.value_out = tf.keras.layers.Dense(1,
                                               name="value_out",
                                               activation=None,
                                               kernel_initializer=normc_initializer(0.01))(last_layer)

        self.base_model = tf.keras.Model(inputs=[image_input, signal_input],
                                         outputs=[(logits_out if logits_out is not None else last_layer),
                                                  self.value_out])

        # self.base_model.summary()

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        """
        (testing)
        input_dict: {'obs': <tf.Tensor 'Const_1:0' shape=(1, 4) dtype=int64>,
                     'obs_flat': <tf.Tensor 'Const_1:0' shape=(1, 4) dtype=int64>}
        (SUMO training: PPO)
                    SampleBatch(0: ['obs', 'new_obs', 'actions', 'prev_actions', 'rewards', 'prev_rewards', 'dones',
                                    'eps_id', 'unroll_id', 'agent_index', 't', 'vf_preds', 'action_dist_inputs',
                                    'action_logp', 'action_prob', 'value_targets', 'advantages', 'obs_flat'])
        """
        logits_out, self.value_out = self.base_model([input_dict['obs']["image"], input_dict['obs']["signal"]])
        return logits_out, state

    def value_function(self) -> TensorType:
        return tf.reshape(self.value_out, [-1])


if __name__ == '__main__':
    random_input = (np.random.rand(64, 128, 3), np.array([2]))
    model = CNNModelLSTMSignal(obs_space=random_input,
                               action_space={0, 1},
                               num_outputs=1,
                               model_config={},
                               name='test')
