"""
Based on stable_baselines3 (version 2.2.1) > dqn > dqn.py;
adjusted section labeled by "### THIS IS ..."
"""

import numpy as np
import stable_baselines3 as sb3
import torch as th
from torch.nn import functional as F


class DoubleDQN(sb3.DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)

                ### THIS IS THE PART THAT IS CHANGED COMPARED TO THE ORIGINAL SB3 (2.2.1) DQN CLASS
                ### based on https://github.com/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/dqn_sb3.ipynb
                # Compute the next Q-values using the online q net
                next_q_values_online = self.q_net(replay_data.next_observations)
                # Select actions as the ones with the highest Q-value
                next_actions_online = th.argmax(next_q_values_online, dim=1, keepdim=True)
                # Estimate the Q-values for the selected actions using the target q network
                next_q_values = th.gather(next_q_values, dim=1, index=next_actions_online)
                ### END OF CHANGES

                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
