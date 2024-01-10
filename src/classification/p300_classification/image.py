import pygame
import random
import datetime

# Initialize Pygame
pygame.init()

# Specify the window dimensions
width, height = 3200, 1692

# Create a Pygame window
screen = pygame.display.set_mode((width, height))

running = True

# Timer to change the image every 500 ms
image_change_timer = pygame.time.set_timer(pygame.USEREVENT, 500)

with open("white_time.txt", "a") as file:
    start_time = datetime.datetime.now()
    file.write(f"\nRecording start at: {start_time}\n")

prev_is_black = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.USEREVENT:
            # Randomly decide whether to display black or white
            # if previous is white, current must be black
            if prev_is_black:
                is_black = random.random() < 0.8  # 80% black, 20% white
                prev_is_black = is_black
            else:
                is_black = True
                prev_is_black = True

            # Create a black or white surface to display
            if is_black:
                color = (0, 0, 0)  # Black (R, G, B)
            else:
                color = (255, 255, 255)  # White (R, G, B)

                current_time = datetime.datetime.now()
                with open("white_time.txt", "a") as file:
                    file.write(f"White image shown at: {current_time - start_time}\n")

            image_surface = pygame.Surface((width, height))
            image_surface.fill(color)

            # Display the black or white surface on the screen
            screen.blit(image_surface, (0, 0))
            pygame.display.flip()

# Quit Pygame
pygame.quit()
