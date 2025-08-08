import os
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks.render_eval_callback import RenderEvalCallback


N_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
MODEL_PATH = "models/ppo_humanoid_final"
BEST_MODEL_PATH = MODEL_PATH + "_best/best_model.zip"
LOG_DIR = "./tensorboard/"
CHECKPOINT_DIR = "./checkpoints/"
VIDEO_DIR = "./videos/"
EVAL_FREQ = 50_000
CHECKPOINT_FREQ = 1_000_000
ENV_ID = "Humanoid-v5"


def make_env(rank=0):
    def _init():
        print(f"Environnement {rank} initialisé")
        env = gym.make(ENV_ID)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    if N_ENVS < 1:
        raise ValueError("N_ENVS doit être >= 1")
    os.makedirs("models", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    eval_env = SubprocVecEnv([make_env(999)])

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // N_ENVS,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_humanoid",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )

    eval_callback = RenderEvalCallback(
        eval_env=eval_env,
        best_model_save_path=MODEL_PATH + "_best/",
        log_path=LOG_DIR + "eval_logs/",
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        verbose=1,
        video_folder=VIDEO_DIR,
        video_length=1000,
    )

    best_model_file = Path(BEST_MODEL_PATH)
    standard_model_file = Path(f"{MODEL_PATH}.zip")

    try:
        if best_model_file.exists():
            print("🔁 Chargement du meilleur modèle précédent...")
            model = PPO.load(best_model_file, env=env, device="auto")
        elif standard_model_file.exists():
            print("🔁 Chargement du modèle standard...")
            model = PPO.load(standard_model_file, env=env, device="auto")
        else:
            print("🚀 Création d’un nouveau modèle PPO...")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                n_steps=2048,
                batch_size=2048,
                n_epochs=5,
                gamma=0.9515020449840818,
                gae_lambda=0.8684939573805941,
                learning_rate=0.00012497886148674835,
                ent_coef=0.0051746157249557305,
                clip_range=0.2928094990990618,
                vf_coef=0.5,
                tensorboard_log=LOG_DIR,
                device="auto"
            )

    except Exception as e:
        print(f"❌ Erreur lors du chargement/création du modèle : {e}")
        exit(1)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save(MODEL_PATH)
    print(f"💾 Modèle final sauvegardé dans {MODEL_PATH}.zip")

    env.close()
    eval_env.close()
