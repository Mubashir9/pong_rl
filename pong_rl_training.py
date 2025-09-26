from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from pong_gym_environment import PongEnv  

def train_pong_agent():
    """Train a PPO agent to play Pong"""
    
    # Create directories for saving models and logs
    models_dir = "models/pong_ppo"
    logs_dir = "logs/pong_ppo"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create the training environment (headless for speed)
    env = make_vec_env(lambda: PongEnv(), n_envs=4)  # 4 parallel environments for faster training
    
    # Create evaluation environment (with rendering for monitoring)
    eval_env = PongEnv(render_mode=None)  # Set to "human" if you want to watch
    
    # Stop training when the agent reaches average reward of 3 (winning most games)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=3, verbose=1)
    
    # Evaluate the agent periodically
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        eval_freq=10000,  # Evaluate every 10k steps
        best_model_save_path=models_dir,
        log_path=logs_dir,
        verbose=1
    )
    
    # Create the PPO agent
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=logs_dir
    )
    
    print("Starting training...")
    print("This may take 10-30 minutes depending on your hardware")
    print("You can monitor progress in tensorboard: tensorboard --logdir logs/")
    
    # Train the agent
    total_timesteps = 500000  # Adjust based on how long you want to train
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    final_model_path = f"{models_dir}/pong_final_model"
    model.save(final_model_path)
    print(f"Training completed! Final model saved to: {final_model_path}")
    
    return model

def test_trained_agent(model_path):
    """Test a trained agent"""
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment with rendering
    env = PongEnv(render_mode="human")
    
    print("Testing trained agent... Press Ctrl+C to stop")
    
    # Test for multiple episodes
    for episode in range(10):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        
        while True:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Render the game
            env.render()
            
            if reward != 0:
                print(f"  Step {steps}: Reward = {reward}")
            
            if done:
                print(f"  Episode finished! Total reward: {total_reward}, Steps: {steps}")
                break
    
    env.close()

def quick_training_demo():
    """Quick training demo with fewer timesteps for testing"""
    print("Running quick training demo (50k timesteps)...")
    
    env = make_vec_env(lambda: PongEnv(), n_envs=2)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        verbose=1
    )
    
    # Quick training
    model.learn(total_timesteps=50000, progress_bar=True)
    
    # Test the partially trained agent
    print("\nTesting partially trained agent...")
    test_env = PongEnv(render_mode="human")
    
    for episode in range(3):
        obs, _ = test_env.reset()
        total_reward = 0
        
        for _ in range(1000):  # Max steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            test_env.render()
            
            if done:
                break
        
        print(f"Episode {episode + 1} reward: {total_reward}")
    
    test_env.close()

if __name__ == "__main__":
    import sys
    
    print("Pong RL Training")
    print("1. Full training (500k timesteps, ~20-30 minutes)")
    print("2. Quick demo (50k timesteps, ~5 minutes)")
    print("3. Test existing model")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        model = train_pong_agent()
        print("\nWould you like to test the trained agent? (y/n)")
        if input().lower().startswith('y'):
            test_trained_agent("models/pong_ppo/best_model")
            
    elif choice == "2":
        quick_training_demo()
        
    elif choice == "3":
        model_path = input("Enter model path (or press Enter for default): ").strip()
        if not model_path:
            model_path = "models/pong_ppo/best_model"
        test_trained_agent(model_path)
        
    else:
        print("Invalid choice")
