import gymnasium as gym
from stable_baselines3 import PPO
from pathlib import Path
import time

MODEL_PATH = "models/ppo_humanoid_final"
BEST_MODEL_PATH = MODEL_PATH + "_best/best_model.zip"
STANDARD_MODEL_PATH = MODEL_PATH + ".zip"

env = gym.make("Humanoid-v5", render_mode="human")

if Path(BEST_MODEL_PATH).exists():
    print("🔁 Chargement du meilleur modèle entraîné...")
    model = PPO.load(BEST_MODEL_PATH)
elif Path(STANDARD_MODEL_PATH).exists():
    print("🔁 Chargement du modèle standard...")
    model = PPO.load(STANDARD_MODEL_PATH)
else:
    raise FileNotFoundError("❌ Aucun modèle trouvé. Entraîne d'abord un modèle avec train.py")

n_episodes = 10

for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    print(f"\n🎮 Épisode {episode + 1} / {n_episodes}")

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        time.sleep(1 / 60)

    print(f"✅ Épisode terminé en {step} étapes | Récompense totale : {total_reward:.2f}")

env.close()
print("🏁 Tous les épisodes sont terminés.")
