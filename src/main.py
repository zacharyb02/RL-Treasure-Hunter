from stable_baselines3 import DQN
from environment import GridWorldEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
from moviepy import ImageSequenceClip

def main():
    # Create the environment
    env = GridWorldEnv(grid_size=10)

    # Initialize the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=8,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        verbose=1,
    )

    # Train the model
    print("Training the agent...")
    model.learn(total_timesteps=100000)

    # Evaluate the model
    print("Evaluating the agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} \u00b1 {std_reward}")

    print("Saving the model...")
    model.save("model/dqn_agent")

    # Test the trained agent
    print("Testing the agent...")
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()  # Visualize the environment

if __name__ == "__main__":
    main()