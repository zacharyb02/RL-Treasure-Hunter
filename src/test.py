from moviepy import ImageSequenceClip
import matplotlib.pyplot as plt
import os
import shutil
from environment import GridWorldEnv
from stable_baselines3 import DQN

def create_video_from_frames(output_dir="frames", video_path="video/agent_performance.mp4", fps=10):
    # List all frame files in order
    frame_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith('.png')]
    
    # Create a video clip from the image sequence
    clip = ImageSequenceClip(frame_files, fps=fps)
    clip.write_videofile(video_path, codec="libx264")
    print(f"Video saved at {video_path}")

def test_and_capture_frames(model, env, output_dir="frames"):
    # Clean up and recreate output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    obs = env.reset()
    done = False
    frame_idx = 0

    while not done:
        # Predict action using the trained model
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Render the current frame
        fig = env.render_image()  # Using the original render_image method
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        fig.savefig(frame_path, bbox_inches='tight', dpi=100)
        plt.close(fig)  # Close the figure properly
        frame_idx += 1

    print(f"Frames captured: {frame_idx}")

def test():
    env = GridWorldEnv(grid_size=10)
    model = DQN.load("model/dqn_agent")
    print("Testing the agent...")
    output_dir = "frames"
    test_and_capture_frames(model, env, output_dir)

    # Create a video from the captured frames
    video_path = "video/agent_performance.mp4"
    print("Creating a video from captured frames...")
    create_video_from_frames(output_dir, video_path, fps=5)  # Reduced fps for better visibility
