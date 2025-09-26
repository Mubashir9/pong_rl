import pygame
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

class Paddle:
    COLOR = (0, 0, 0)  # BLACK
    VEL = 4

    def __init__(self, x, y, width, height):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height

    def draw(self, win):
        pygame.draw.rect(win, self.COLOR, (self.x, self.y, self.width, self.height))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def move_by_action(self, action, height):
        """Move paddle based on action number with boundary checking"""
        if action == 1 and self.y - self.VEL >= 0:  # Up
            self.y -= self.VEL
        elif action == 2 and self.y + self.VEL + self.height <= height:  # Down
            self.y += self.VEL
        # action == 0 means stay (do nothing)

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

class Ball:
    MAX_VEL = 5
    COLOR = (0, 0, 0)  # BLACK
    
    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = self.MAX_VEL
        self.y_vel = 0
        
    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (int(self.x), int(self.y)), self.radius)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.y_vel = 0
        self.x_vel *= -1

class PongEnv(gym.Env):
    """Custom Pong Environment that follows gym interface"""
    
    def __init__(self, render_mode=None):
        super(PongEnv, self).__init__()
        
        # Game constants
        self.WIDTH = 700
        self.HEIGHT = 500
        self.PADDLE_WIDTH = 20
        self.PADDLE_HEIGHT = 100
        self.BALL_RADIUS = 7
        self.FPS = 60
        self.MAX_SCORE = 5  
        
        # Define action and observation space
        # Action space: 0=Stay, 1=Up, 2=Down
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [ball_x, ball_y, ball_x_vel, ball_y_vel, agent_paddle_y, opponent_paddle_y]
        # Normalized to [0, 1] range
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )
        
        # Initialize pygame if rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Pong RL Training")
            self.clock = pygame.time.Clock()
        
        # Initialize game objects
        self._initialize_game()
    
    def _initialize_game(self):
        """Initialize game objects"""
        self.left_paddle = Paddle(10, self.HEIGHT//2 - self.PADDLE_HEIGHT//2, 
                                 self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.right_paddle = Paddle(self.WIDTH - 10 - self.PADDLE_WIDTH, 
                                  self.HEIGHT//2 - self.PADDLE_HEIGHT//2,
                                  self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.ball = Ball(self.WIDTH//2, self.HEIGHT//2, self.BALL_RADIUS)
        
        self.left_score = 0
        self.right_score = 0
    
    def _get_obs(self):
        """Get current observation (normalized to [0,1])"""
        obs = np.array([
            self.ball.x / self.WIDTH,                    # Ball x position
            self.ball.y / self.HEIGHT,                   # Ball y position  
            (self.ball.x_vel + self.ball.MAX_VEL) / (2 * self.ball.MAX_VEL),  # Ball x velocity
            (self.ball.y_vel + self.ball.MAX_VEL) / (2 * self.ball.MAX_VEL),  # Ball y velocity
            self.left_paddle.y / (self.HEIGHT - self.PADDLE_HEIGHT),   # Agent paddle position
            self.right_paddle.y / (self.HEIGHT - self.PADDLE_HEIGHT),  # Opponent paddle position
        ], dtype=np.float32)
        return obs
    
    def _handle_collision(self):
        """Handle ball collisions (same as your original logic)"""
        ball = self.ball
        left_paddle = self.left_paddle
        right_paddle = self.right_paddle
        
        # Top/bottom wall collision
        if ball.y + ball.radius >= self.HEIGHT:
            ball.y_vel *= -1
        elif ball.y - ball.radius <= 0:
            ball.y_vel *= -1

        # Paddle collisions
        if ball.x_vel < 0:  # Moving left
            if (ball.y >= left_paddle.y and ball.y <= left_paddle.y + left_paddle.height and
                ball.x - ball.radius <= left_paddle.x + left_paddle.width):
                ball.x_vel *= -1
                middle_y = left_paddle.y + left_paddle.height // 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (left_paddle.height / 2) / ball.MAX_VEL
                ball.y_vel = -1 * (difference_in_y / reduction_factor)
                
                # Increase ball speed after paddle hit
                self._increase_ball_speed()
                
        else:  # Moving right
            if (ball.y >= right_paddle.y and ball.y <= right_paddle.y + right_paddle.height and
                ball.x + ball.radius >= right_paddle.x):
                ball.x_vel *= -1
                middle_y = right_paddle.y + right_paddle.height // 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (right_paddle.height / 2) / ball.MAX_VEL
                ball.y_vel = -1 * (difference_in_y / reduction_factor)
                
                # Increase ball speed after paddle hit
                self._increase_ball_speed()
    
    def _increase_ball_speed(self):
        """Increase ball speed slightly after each paddle hit"""
        speed_increase = 1.05  # 5% increase each hit
        max_speed = self.ball.MAX_VEL * 2  # Cap at 2x original speed
        
        # Increase x velocity (maintain direction)
        if self.ball.x_vel > 0:
            self.ball.x_vel = min(self.ball.x_vel * speed_increase, max_speed)
        else:
            self.ball.x_vel = max(self.ball.x_vel * speed_increase, -max_speed)
        
        # Increase y velocity (maintain direction)  
        if self.ball.y_vel > 0:
            self.ball.y_vel = min(self.ball.y_vel * speed_increase, max_speed)
        elif self.ball.y_vel < 0:
            self.ball.y_vel = max(self.ball.y_vel * speed_increase, -max_speed)
    
    def _simple_ai_opponent(self):
        """Simple AI for opponent (similar to your original)"""
        paddle_center = self.right_paddle.y + self.right_paddle.height // 2
        ball_y = self.ball.y
        
        reaction_distance = self.WIDTH * 0.6
        
        if self.ball.x < reaction_distance:
            target_y = self.HEIGHT // 2
            dead_zone = 30
        else:
            if self.ball.x_vel > 0:
                prediction_error = random.randint(-20, 20)
                target_y = ball_y + prediction_error
            else:
                target_y = self.HEIGHT // 2
            dead_zone = 15
        
        ai_speed = 3
        
        if paddle_center < target_y - dead_zone:
            if self.right_paddle.y + ai_speed + self.right_paddle.height <= self.HEIGHT:
                self.right_paddle.y += ai_speed
        elif paddle_center > target_y + dead_zone:
            if self.right_paddle.y - ai_speed >= 0:
                self.right_paddle.y -= ai_speed
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._initialize_game()
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Move agent paddle based on action
        self.left_paddle.move_by_action(action, self.HEIGHT)
        
        # Move opponent paddle (simple AI)
        self._simple_ai_opponent()
        
        # Move ball
        self.ball.move()
        
        # Handle collisions
        self._handle_collision()
        
        # Check for scoring
        reward = 0
        done = False
        
        if self.ball.x < 0:  # Right player (opponent) scores
            self.right_score += 1
            reward = -1
            self.ball.reset()
        elif self.ball.x > self.WIDTH:  # Left player (agent) scores
            self.left_score += 1
            reward = 1
            self.ball.reset()
        
        # Check if game is over
        if self.left_score >= self.MAX_SCORE or self.right_score >= self.MAX_SCORE:
            done = True
        
        return self._get_obs(), reward, done, False, {}
    
    def render(self, mode="human"):
        """Render the environment"""
        if mode == "human" and self.screen is not None:
            # Handle pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # Signal to close
            
            # Fill screen
            self.screen.fill((255, 255, 255))  # White background
            
            # Draw paddles
            self.left_paddle.draw(self.screen)
            self.right_paddle.draw(self.screen)
            
            # Draw ball
            self.ball.draw(self.screen)
            
            # Draw center line
            for i in range(10, self.HEIGHT, self.HEIGHT//20):
                if i % 2 == 1:
                    continue
                pygame.draw.rect(self.screen, (0, 0, 0), 
                               (self.WIDTH//2 - 5, i, 10, self.HEIGHT//20))
            
            # Draw scores
            if hasattr(self, 'left_score') and hasattr(self, 'right_score'):
                font = pygame.font.Font(None, 74)
                left_text = font.render(str(self.left_score), True, (0, 0, 0))
                right_text = font.render(str(self.right_score), True, (0, 0, 0))
                self.screen.blit(left_text, (self.WIDTH//4 - left_text.get_width()//2, 20))
                self.screen.blit(right_text, (3*self.WIDTH//4 - right_text.get_width()//2, 20))
            
            pygame.display.flip()
            
            # Control frame rate
            if self.clock:
                self.clock.tick(60)
            
            return True
        return True
    
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.quit()

# Test the environment
if __name__ == "__main__":
    env = PongEnv(render_mode="human")
    obs, _ = env.reset()
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation:", obs)
    
    # Test with random actions
    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        if reward != 0:
            print(f"Reward: {reward}")
        
        if done:
            print("Game over! Resetting...")
            obs, _ = env.reset()
    
    env.close()