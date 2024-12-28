import numpy as np
import matplotlib.pyplot as plt

def plot_save_results(log_folder):
    # Load evaluation results
    data = np.load(f'{log_folder}/evaluations.npz')
    timesteps = data['timesteps']
    results = data['results'].mean(axis=1)  # Assuming results has shape (n_evaluations, n_episodes)

    # Plot the episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, results)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Episode Rewards Over Time')
    plt.grid(True)
    
    # Save the figure
    plt.savefig('episode_rewards.png')
    plt.close()