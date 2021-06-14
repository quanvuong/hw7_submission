import sys
from pprint import pprint
import os
from jedi.plugins import stdlib
import numpy as np
import gym
from numpy.core.fromnumeric import size
import sapien_rl.env
import copy
import argparse


class MPC:
    def __init__(self, env, plan_horizon=4, popsize=200, num_elites=20, max_iters=4, use_mpc=True):
        """
        :param env:
        :param plan_horizon: 
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param use_mpc: Whether to use only the first action of a planned trajectory
        """
        self.env = env
        self.use_mpc = use_mpc
        self.plan_horizon = plan_horizon
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites
        self.action_dim = env.action_space.shape[0]
        # used to clip your actions
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.reset()

    def reset(self):
        self.mean = np.zeros((self.plan_horizon * self.action_dim))
        self.std = 0.5 * np.ones((self.plan_horizon * self.action_dim))

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        next_states = []
        rewards = []
        for i in range(len(states)):
            next_state, reward, info = self.env.mpc_step(states[i], actions[i])
            next_states.append(next_state)
            rewards.append(reward)
        return next_states, rewards

    def cem_optimize(self, state):
        mean = self.mean.copy()
        std = self.std.copy()
        initial_state = state.copy()

        min_std = 0.01

        for i in range(self.max_iters):

            # s (self.popsize, state size) after these 2 lines
            s = initial_state[None, :]
            s = np.repeat(s, self.popsize, axis=0)

            # ensure std is always of a certain minimum value
            std[std < min_std] = min_std

            # sample actions from mean and std
            a = np.random.normal(
                mean,
                std,
                size=(
                    self.popsize,
                    self.plan_horizon * self.action_dim
                )
            )

            # used to keep track of the rewards of all state-action pair
            r = np.empty((self.popsize, self.plan_horizon))

            # Roll out trajectories
            for t in range(self.plan_horizon):

                ad = self.action_dim

                actions_to_take = a[
                    :,  # indexing across the population
                    # selecting the t-th action in the sampled action
                    t*ad:(t+1)*ad
                ]

                s, sa_r = self.predict_next_state_gt(
                    s, actions_to_take
                )

                # keep track of the reward of each state-action pair
                # of the population
                r[:, t] = sa_r

            # Compute the sum of reward of each sample in the population
            sor = np.sum(r, axis=1)

            # Find the elites and update mean and std
            elites = a[
                np.argsort(sor)  # ascending sort
            ][-self.num_elites:]

            mean = np.mean(elites, axis=0)
            std = np.std(elites, axis=0)

        return mean, std

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        if self.use_mpc == False:
            if t % self.plan_horizon == 0:
                # TODO
                pass
            else:
                # TODO
                pass
            return np.zeros(8)  # TODO
        else:

            m, s = self.cem_optimize(state)

            # Pick the first action to take
            a = m[:self.action_dim]

            # Remove the first action from mean and std
            m = m[self.action_dim:]
            s = s[self.action_dim:]

            # To ensure mean and std contains plan_horizon
            # actions, initialize the mean and std
            # of the action at the end of the plan_horizon
            m = np.concatenate((m, np.zeros(self.action_dim)))
            s = np.concatenate((s, 0.5 * np.ones(self.action_dim)))

            self.mean = m
            self.std = s

            return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0, help='save index')
    opt = parser.parse_args()

    env = gym.make('OpenCabinet_state_45267_link_0-v4')
    env.reset(level=3)
    env.perturb_gripper()
    obs = env.get_obs()

    cem_mpc = MPC(env)
    states = []
    actions = []
    next_states = []

    # env.render('human')
    for t in range(150):
        state = env.get_state("world")
        action = cem_mpc.act(state, t)
        states.append(state)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        next_states.append(obs)
        print("---step #%d----" % t)
        pprint(info)
        print("reward: %f" % reward, info['eval_info'])
        if info['eval_info']['success']:
            print("Bravo! Open the door!")
            break
        if done:
            break
        sys.stdout.flush()

    # save the transitions
    np.save("trajectories/%d_traj.npy" % opt.id,
            {"states": states, "actions": actions, "next_states": next_states})
