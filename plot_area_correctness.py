import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# log_file = 'logs/2:10PM-05/rewards.log'
# log_file = "rewards_rotated.log"
log_file = "rewards_area_range.log"

area_correctness = []
steps = []

with open(log_file, 'r') as f:
    step = 0
    for line in f:
        area_correctness_match = re.search(r'Mean area correctness: ([0-9eE\.+-]+)', line)
        if area_correctness_match:
            area_correctness.append(float(area_correctness_match.group(1)))
            steps.append(step)
            step += 1

window = 20
if len(area_correctness) >= window:
    ma = np.convolve(area_correctness, np.ones(window)/window, mode='valid')
    ma_steps = steps[window-1:len(area_correctness)]
else:
    ma = []
    ma_steps = []

plt.figure(figsize=(10, 5))
plt.plot(steps, area_correctness, label='Area Correctness Reward', alpha=0.5)
if len(ma_steps):
    plt.plot(ma_steps, ma, label=f'Moving Average (window={window})', color='red', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Area Correctness Reward')
plt.title('Area Correctness Reward over Training Steps')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.5))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.tight_layout()
plt.show() 