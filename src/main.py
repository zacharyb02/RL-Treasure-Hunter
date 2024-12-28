from stable_baselines3 import DQN
from environment import GridWorldEnv
from plot import plot_save_results
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
from moviepy import ImageSequenceClip

def main():
    # Create the environment
    env = GridWorldEnv(grid_size=10)

    # Initialize the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,         
        buffer_size=200000,            
        learning_starts=5000,          
        batch_size=128,               
        gamma=0.98,                    
        train_freq=4,                  
        target_update_interval=1000,   
        exploration_fraction=0.2,      
        exploration_final_eps=0.05,    
        policy_kwargs=dict(net_arch=[256, 256]),        
        verbose=1
    )

    # Callback for evaluation
    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./logs/',
        log_path='./logs/', 
        eval_freq=500,
        deterministic=True, 
        render=False
    )

    # Train the model
    print("Training the agent...")
    model.learn(total_timesteps=100000,log_interval=100, callback=eval_callback)

    # Evaluate the model
    print("Evaluating the agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} \u00b1 {std_reward}")

    # Plot and save the evaluation results
    plot_save_results('./logs')

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