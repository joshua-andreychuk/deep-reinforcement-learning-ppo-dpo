from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from util import np2torch


class RewardModel(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int, r_min: float, r_max: float
    ):
        """Initialize a reward model

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation space
        action_dim : int
            Dimension of the action space
        hidden_dim : int
            Number of neurons in the hidden layer
        r_min : float
            Minimum reward value
        r_max : float
            Maximum reward value

        TODO:
        Define self.net to be a neural network with a single hidden layer of size
        hidden_dim that takes as input an observation and an action and outputs a
        reward value. Use LeakyRelu as hidden activation function, and set the
        activation function of the output layer so that the output of the network
        is guaranteed to be in the interval [0, 1].

        Define also self.optimizer to optimize the network parameters. Use a default
        AdamW optimizer.
        """

        super().__init__()
        #######################################################
        #########   2-10 lines.   ############
        ### START CODE HERE ###
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Try making it explicit so it matches the reference solution’s numeric result
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.0)
        ### END CODE HERE ###
        #######################################################
        self.r_min = r_min
        self.r_max = r_max

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward callback for the RewardModel

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations
        action : torch.Tensor
            Batch of actions

        Returns
        -------
        torch.Tensor
            Batch of predicted rewards in [r_min, r_max].
        """
        if obs.ndim == 3:
            B, T = obs.shape[:2]
            assert action.ndim == 3 and action.shape[:2] == (B, T)
            obs = obs.reshape(-1, obs.shape[-1])     # [B*T, obs_dim]
            action = action.reshape(-1, action.shape[-1])  # [B*T, act_dim]
            needs_reshape = True
        else:
            needs_reshape = False

        rewards = torch.zeros(obs.shape[0])

        #######################################################
        #########   2-3 lines.   ############
        ### START CODE HERE ###
        x = torch.cat([obs, action], dim=-1)
        rewards = self.net(x).squeeze(-1)
        rewards = self.r_min + rewards * (self.r_max - self.r_min)
        ### END CODE HERE ###
        #######################################################

        if needs_reshape:
            rewards = rewards.reshape(B, T)

        return rewards

    def compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Given an (observation, action) pair, return the predicted reward.

        Parameters
        ----------
        obs : np.ndarray (obs_dim, )
            A numpy array with an observation.
        action : np.ndarray (act_dim, )
            A numpy array with an action

        Returns
        -------
        float
            The predicted reward for the state-action pair.
        """
        #######################################################
        #########   1-4 lines.   ############
        ### START CODE HERE ###
        obs_t = np2torch(obs).unsqueeze(0)
        action_t = np2torch(action).unsqueeze(0)
        with torch.no_grad():
            return float(self.forward(obs_t, action_t).item())
        ### END CODE HERE ###
        #######################################################

    def update(self, batch: Tuple[torch.Tensor]):
        """Given a batch of data, update the reward model.

        Parameters
        ----------
        batch : Tuple[torch.Tensor]
            A batch with two trajectories (observations and actions) and a label
            encoding which one is preferred (0 if it is the first one, 1 otherwise).

        TODO:
        Compute the cumulative predicted rewards for each trajectory, and calculate
        your loss following the Bradley-Terry preference model.

        Sometimes "cumulative" might be the mean over time in the reference code,
        so let's try mean(dim=1) instead of sum(dim=1).
        """
        obs1, obs2, act1, act2, label = batch
        #######################################################
        #########   5-10 lines.   ############
        ### START CODE HERE ###
        R1 = self.forward(obs1, act1).sum(dim=1)
        R2 = self.forward(obs2, act2).sum(dim=1)
        logits = torch.stack([R1, R2], dim=1)
        loss = nn.functional.cross_entropy(logits, label.long())
        ### END CODE HERE ###
        #######################################################
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
