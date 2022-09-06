import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import TensorDict


class CombinedExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        super(CombinedExtractor, self).__init__(observation_space, features_dim=1)
        # We assume CxHxW images (channels first)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "rgb":
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                # hidden_dims = 3
                # output_dims = 1
                # input_dims = subspace.shape[1]
                # print("--------------------------------------")
                # print(input_dims)
                # print(subspace.shape)
                # print("--------------------------------------")
                # extractors[key] = nn.Sequential(
                #     nn.Linear(input_dims, hidden_dims),
                #     nn.ReLU(),
                #     nn.Linear(hidden_dims, output_dims),
                #     nn.ReLU(),
                #     nn.Flatten(),
                # )
                total_concat_size += get_flattened_obs_dim(subspace)
                # total_concat_size += output_dims
                # print("--------------------------------------")
                # print(total_concat_size)
                # print("--------------------------------------")

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


def combined_extractor(config):
    kwargs = {}
    kwargs["policy"] = "MultiInputLstmPolicy"
    kwargs["policy_kwargs"] = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256),
        net_arch=[],
        n_lstm_layers = 1,
        lstm_hidden_size = 256,
        activation_fn = nn.ReLU,
        )
    kwargs["target_kl"] = 0.1
    return kwargs
