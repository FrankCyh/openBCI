import pygame

### READ GROUND TRUTH from txt file and STORE IN A LIST gt
file_path = 'inference_on_leting_data_only-gt.txt'
gt = []
with open(file_path, 'r') as file:
    for line in file:
        gt.append(int(line.strip()))
# only take first 30s, i.e. first 60 labels
gt = gt[:60]

### CREATE WINDOW

# Initialize Pygame
pygame.init()

# Specify the window dimensions
width, height = 480, 360

# Create a Pygame window
screen = pygame.display.set_mode((width, height))

running = True

# Timer to change the image every 500 ms
image_change_timer = pygame.time.set_timer(pygame.USEREVENT, 500)

i = 0
stop = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.USEREVENT:

            # Create a black or white surface to display according to gt
            if gt[i] == 0:
                color = (0, 0, 0)  # Black (R, G, B)
            else:
                color = (255, 255, 255)  # White (R, G, B)

            image_surface = pygame.Surface((width, height))
            image_surface.fill(color)

            # Display the black or white surface on the screen
            screen.blit(image_surface, (0, 0))
            pygame.display.flip()

            i += 1
            if i == len(gt):
                stop = True
                break
    if stop:
        break

# Quit Pygame
pygame.quit()
