import pygame
import cv2
from stable_baselines3 import PPO
from pong_gym_environment import PongEnv

# Load model and create environment
model = PPO.load("models/pong_ppo/best_model")
env = PongEnv(render_mode="human")

# Setup video recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('pong_demo.mp4', fourcc, 30, (700, 500))

print("Recording 30 seconds...")

# Record for 30 seconds at 30 FPS = 900 frames
obs, _ = env.reset()
for frame in range(2400):
    # Get action and step
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    # Render and capture
    env.render()
    screen = pygame.display.get_surface()
    frame_array = pygame.surfarray.array3d(screen).swapaxes(0, 1)
    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)
    
    if done:
        obs, _ = env.reset()

# Cleanup
video_writer.release()
env.close()
print("Video saved as: pong_demo.mp4")