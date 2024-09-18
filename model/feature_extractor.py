from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import torch


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_ts, hidden_size_lstm, num_layers_lstm):
        """

        :param spaces.Dict observation_space: observation space of the environment
        :param int n_ts: number of time series features
        :param int hidden_size_lstm: size of the hidden state of the LSTM
        :param int num_layers_lstm: number of LSTM layers
        """
        # Features_dim just a dummy value as we don't know the dimension at this point
        super().__init__(observation_space, features_dim=1)

        # Initializations
        self.observation_space = observation_space
        extractors = {}
        total_concat_size = hidden_size_lstm
        len_ts = None

        # Go over all the spaces and compute the output feature sizes
        for key, subspace in self.observation_space.spaces.items():
            if "hist" in key:
                if len_ts is not None:
                    assert get_flattened_obs_dim(subspace) == len_ts, "Please ensure that the histories have the same lengths."
                else:
                    len_ts = get_flattened_obs_dim(subspace)
            else:  # not a time series
                extractors[key] = torch.nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
        extractors["time_series"] = torch.nn.LSTM(input_size=n_ts, hidden_size=hidden_size_lstm, num_layers=num_layers_lstm, batch_first=True)
        self.extractors = torch.nn.ModuleDict(extractors)

        # Update the features dim
        self._features_dim = total_concat_size

        return

    def forward(self, observations):
        """

        :param dict observations: dictionary with batch of observations
        :return: feature tensor of size (batch_size, self._features_dim)
        :rtype: torch.Tensor
        """
        list_features = []
        list_ts = []  # time series data

        # self.extractors contains torch.nn.Modules that do all the processing
        for key, subspace in sorted(self.observation_space.spaces.items()):
            if "hist" in key:
                list_ts.append(observations[key])
            else:  # not a time series
                list_features.append(self.extractors[key](observations[key]))

        # Pass the input through the LSTM layer
        # The initial hidden state and the initial cell state default to zeros if (h_0, c_0) is not provided
        _, (h_n, _) = self.extractors["time_series"](torch.cat([ts.unsqueeze(dim=2) for ts in list_ts], dim=2))
        # Return the final hidden state as the extracted features
        list_features.append(h_n[-1])

        # Return a (batch_size, self._features_dim) PyTorch tensor
        return torch.cat(list_features, dim=1)
