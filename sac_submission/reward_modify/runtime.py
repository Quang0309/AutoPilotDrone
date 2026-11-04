"""Environment wrappers used by the SAC navigation model."""
import io
from contextlib import contextmanager, redirect_stdout
import numpy as np
import gymnasium as gym
import torch
from gym_pybullet_drones.envs.NavigateAviary import NavigateAviary
from gym_pybullet_drones.utils.NavigateExtractor import NavigateExtractor
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


class ResetPID(gym.Wrapper):
    """Reset the PID controllers at the start of each episode."""
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        for controller in self.unwrapped.ctrl:
            controller.reset()
        return observation, info


class RelabelFailureAsTerminated(gym.Wrapper):
    """Separate failures from the time limit."""
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        base = self.unwrapped
        state = base._getDroneStateVector(0)
        info = dict(info)
        collision = bool(info.get('collision', False))
        out_of_bounds = bool(abs(state[0]) > base.BOUND_LIMIT_XY or
                             abs(state[1]) > base.BOUND_LIMIT_XY or
                             state[2] > 2.0 or state[2] < 0.15)
        excessive_tilt = bool(abs(state[7]) > 1.5 or abs(state[8]) > 1.5)
        reason = info.get('truncation_reason')
        failure = collision or out_of_bounds or excessive_tilt or reason in {
            'collision', 'out_of_bounds', 'flipped'}
        raw_terminated, raw_truncated = bool(terminated), bool(truncated)
        terminated = bool(terminated or failure)
        timeout = bool(raw_truncated and reason == 'timeout' and not terminated)
        truncated = timeout
        info.update(collision=collision, out_of_bounds=out_of_bounds,
                    excessive_tilt=excessive_tilt, timeout=timeout,
                    raw_terminated=raw_terminated, raw_truncated=raw_truncated)
        return observation, reward, terminated, truncated, info


class SeededNavigate(gym.Wrapper):
    """Navigation environment with a repeatable random seed."""
    def __init__(self, difficulty, seed, gui=False, record=False):
        outside_state = np.random.get_state()
        np.random.seed(seed)
        try:
            with redirect_stdout(io.StringIO()):
                base = NavigateAviary(difficulty=difficulty, obs=ObservationType.ALL,
                                      act=ActionType.VEL, gui=gui, record=record)
        finally:
            np.random.set_state(outside_state)
        super().__init__(RelabelFailureAsTerminated(ResetPID(base)))
        self.random_state = np.random.RandomState(seed).get_state()

    @contextmanager
    def local_random(self):
        outside_state = np.random.get_state()
        np.random.set_state(self.random_state)
        try:
            yield
        finally:
            self.random_state = np.random.get_state()
            np.random.set_state(outside_state)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.random_state = np.random.RandomState(seed).get_state()
        with self.local_random():
            return self.env.reset(seed=seed, options=options)

    def step(self, action):
        with self.local_random():
            return self.env.step(action)


class SACNavigateExtractor(NavigateExtractor):
    """Feature extractor for depth images and drone state."""
    def __init__(self, observation_space, cnn_output_dim=32):
        super().__init__(observation_space, cnn_output_dim)
        torch.nn.init.kaiming_uniform_(self.cnn_linear.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.cnn_linear.bias, 0.01)
