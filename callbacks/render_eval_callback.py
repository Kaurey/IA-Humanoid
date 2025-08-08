import os
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import RecordVideo

class RenderEvalCallback(EvalCallback):
    def __init__(self, *args, video_folder="videos/", video_length=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_folder = video_folder
        self.video_length = video_length
        os.makedirs(video_folder, exist_ok=True)

    def _record_video(self):
        print(f"🎥 Enregistrement vidéo pour le meilleur modèle à {self.num_timesteps} étapes...")

        raw_env = gym.make("Humanoid-v5", render_mode="rgb_array")
        env = RecordVideo(
            raw_env,
            video_folder=self.video_folder,
            name_prefix=f"best_model_step_{self.num_timesteps}",
            episode_trigger=lambda x: True,
            disable_logger=True
        )

        obs, _ = env.reset()
        for _ in range(self.video_length):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.best_mean_reward is not None and self.last_mean_reward is not None:
            if self.last_mean_reward > self.best_mean_reward:
                self._record_video()
        return result
