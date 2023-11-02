import matplotlib.pyplot as plt
from lab5 import calculate_alpha_beta_ratio
import numpy as np
from matplotlib.animation import FuncAnimation

# Neurofeedback threshold
threshold = 0.7  # Adjust the threshold as needed

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
y_limit = 0.6
ax.set_ylim(0, y_limit)
ax.set_aspect('equal')
ax.xaxis.set_visible(True)
ax.set_xticklabels([])
ax.yaxis.set_visible(False)

character = plt.Rectangle((0.3, 0), 0.1, 0.1, color='red')
ax.add_patch(character)

bonus_square = plt.Rectangle((0.6, 0), 0.1, 0.1, color='cyan')
ax.add_patch(bonus_square)

# Finish line & Starting line
finish_line = plt.Line2D([0, 1], [y_limit - 0.005, y_limit - 0.005], color='green')
ax.add_line(finish_line)

start_line = plt.Line2D([0, 1], [0.005, 0.005], color='olive')
ax.add_line(start_line)
ax.text(0.34, -0.1, 'You', fontsize=10, ha='center', va='center', color='black')
ax.text(0.64, -0.1, 'Competitor', fontsize=10, ha='center', va='center', color='black')


def init():
    character.set_xy((0.3, 0.0))
    bonus_square.set_xy((0.6, 0))
    ax.add_patch(character)
    ax.add_patch(bonus_square)
    return character, bonus_square, finish_line, start_line


def animate(frame):
    alpha_beta_ratio = calculate_alpha_beta_ratio()

    # Move the character based on the alpha/beta ratio
    if alpha_beta_ratio > threshold:
        character.set_y(character.get_y() + 0.05)

    #Always move the competitor a set distance of 0.03
    bonus_square.set_y(bonus_square.get_y() + 0.03)

    if character.get_y() >= y_limit - character.get_height():
        plt.title('Winner! \n You are great at neuromodulation!')
        ani.event_source.stop()
        plt.pause(5)  # Pause for 5 seconds
        quit()
    elif bonus_square.get_y() >= y_limit - bonus_square.get_height():
        plt.title('Loser! \n You can\'t even neuromodulate!')
        ani.event_source.stop()
        plt.pause(5)  # Pause for 5 seconds
        quit()

    return character, bonus_square, finish_line, start_line

ani = FuncAnimation(fig, animate, init_func=init, frames=np.arange(0, 200), blit=True)
plt.show()