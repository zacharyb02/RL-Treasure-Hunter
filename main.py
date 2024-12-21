from stable_baselines3 import DQN
from environment import GridWorldEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
from moviepy import ImageSequenceClip

def create_video_from_frames(output_dir="frames", video_path="agent_performance.mp4", fps=10):
    # List all frame images in order
    frame_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith('.png')]

    # Create a video clip from the image sequence
    clip = ImageSequenceClip(frame_files, fps=fps)
    clip.write_videofile(video_path, codec="libx264")

def test_and_capture(model, env, output_dir="frames"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    obs = env.reset()
    done = False
    frame_idx = 0

    while not done:
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Save the current frame as an image
        fig = env.render_image()
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        fig.savefig(frame_path)
        frame_idx += 1


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

    # Test the trained agent
    print("Testing the agent...")
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()  # Visualize the environment

        # Capture testing frames
    test_and_capture(model, env)

    # Create a video from the frames
    create_video_from_frames()

if __name__ == "__main__":
    main()
