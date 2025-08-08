import optuna
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from pathlib import Path
import os

ENV_ID = "Humanoid-v5"
TOTAL_TIMESTEPS = 200_000
N_TRIALS = 20
LOG_DIR = "./optuna_logs/"
BEST_MODEL_DIR = "./optuna_best_model/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

def optimize_ppo(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 3e-4)
    gamma = trial.suggest_uniform("gamma", 0.95, 0.9999)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 0.99)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-2)
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    n_steps = trial.suggest_int("n_steps", 1024, 4096, step=512)
    batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192, 16384])

    if batch_size > n_steps:
        raise optuna.exceptions.TrialPruned()

    env = make_vec_env(ENV_ID, n_envs=1, wrapper_class=Monitor)

    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=BEST_MODEL_DIR,
                                 log_path=LOG_DIR,
                                 eval_freq=10_000,
                                 deterministic=True,
                                 render=False)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=0,
        tensorboard_log=LOG_DIR
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
        mean_reward, _ = eval_callback.last_mean_reward, None
    except Exception as e:
        print(f"❌ Trial échoué : {e}")
        raise optuna.exceptions.TrialPruned()

    return mean_reward or -9999

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=N_TRIALS, n_jobs=1)

    print("✅ Meilleurs hyperparamètres :")
    print(study.best_params)

    study_path = Path(LOG_DIR) / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        import pickle
        pickle.dump(study, f)
