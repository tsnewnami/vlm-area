import re
import matplotlib.pyplot as plt
import numpy as np

# Log files and their corresponding legend labels
log_files = [
    ("rewards.log", "20-35 Dimension Range"),
    ("rewards_rotated.log", "Rotation"),
    ("rewards_area_range.log", "5-35 Dimension Range"),
]

window = 20

plt.figure(figsize=(10, 5))

for log_file, label in log_files:
    area_correctness = []
    steps = []
    with open(log_file, 'r') as f:
        step = 0
        for line in f:
            match = re.search(r'Mean area correctness: ([0-9eE\.+-]+)', line)
            if match:
                area_correctness.append(float(match.group(1)))
                steps.append(step)
                step += 1
    if len(area_correctness) >= window:
        ma = np.convolve(area_correctness, np.ones(window)/window, mode='valid')
        ma_steps = steps[window-1:len(area_correctness)]
        plt.plot(ma_steps, ma, label=label, linewidth=2)
    else:
        plt.plot(steps, area_correctness, label=label, linewidth=2)

plt.xlabel('Step')
plt.ylabel('Running Average Area Correctness Reward (window=20)')
plt.title('Running Average for Shape Variations over Training Steps')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 