import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from utils import create_video_from_frames

logger = logging.getLogger(__name__)


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, video_save_path, model_save_path, eval_freq, n_eval_episodes,
                 deterministic,
                 verbose=1):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.video_save_path = video_save_path
        self.model_save_path = model_save_path
        self.eval_freq = eval_freq
        self.deterministic = deterministic

        self._best_mean_reward = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            logger.info("Starting evaluation at episode {}".format(self.num_timesteps))
            all_rewards = np.zeros(self.eval_env.num_envs)
            obs = self.eval_env.reset()
            obs_frames = []
            orig_frames = []

            accelerations = []
            used_brake_penalty = 0

            steer_change_rates = []
            used_steer_penalty = 0

            done = np.zeros(self.eval_env.num_envs).astype(bool)

            for _ in range(1000):
                if np.all(done):
                    break
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, infos = self.eval_env.step(action)

                for info in infos:
                    accelerations.append(info['acceleration'])
                    used_brake_penalty += 1 if info['brake_penalty_used'] else 0

                    steer_change_rates.append(info['steer_change_rate'])
                    used_steer_penalty += 1 if info['steer_penalty_used'] else 0

                orig_frames.append(np.array(self.eval_env.get_images()))

                if obs.shape[-1] == 1:
                    obs_frames.append(np.repeat(obs, 3, axis=-1))
                else:
                    obs_frames.append(obs)
                all_rewards += reward

            for i in range(self.eval_env.num_envs):
                obs_video_path = self.video_save_path + f'/obs/{self.num_timesteps:07}/{i:02}_{all_rewards[i]:.1f}.mp4'
                orig_video_path = self.video_save_path + f'/orig/{self.num_timesteps:07}/{i:02}_{all_rewards[i]:.1f}.mp4'
                create_video_from_frames([item[i, ...] for item in obs_frames], obs_video_path)
                create_video_from_frames([item[i, ...] for item in orig_frames], orig_video_path)

            mean_reward = np.sum(all_rewards) / self.eval_env.num_envs

            if self._best_mean_reward is None or mean_reward > self._best_mean_reward:
                model_path = self.model_save_path + f"{self.num_timesteps:07}_{mean_reward:.1f}" + '.zip'
                self.model.save(model_path)
                logger.info(f"Found best model with reward {mean_reward:.1f}, saved to {model_path}")

                self._best_mean_reward = mean_reward

            self.logger.record("eval/acceleration/mean", np.mean(accelerations))
            self.logger.record("eval/acceleration/max", np.max(accelerations))
            self.logger.record("eval/acceleration/min", np.min(accelerations))
            self.logger.record("eval/acceleration/used_brake_penalty", used_brake_penalty)

            self.logger.record("eval/steer_change_rate/mean", np.mean(steer_change_rates))
            self.logger.record("eval/steer_change_rate/max", np.max(steer_change_rates))
            self.logger.record("eval/steer_change_rate/min", np.min(steer_change_rates))
            self.logger.record("eval/steer_change_rate/used_steer_penalty", used_steer_penalty)

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/max_reward", np.max(all_rewards))
            self.logger.record("eval/min_reward", np.min(all_rewards))

        return True
