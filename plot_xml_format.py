import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

log_file = 'logs/2:10PM-05/rewards.log'

xml_format = []
steps = []

with open(log_file, 'r') as f:
    step = 0
    for line in f:
        xml_match = re.search(r'Mean XML format: ([0-9eE\.+-]+)', line)
        if xml_match:
            xml_format.append(float(xml_match.group(1)))
            steps.append(step)
            step += 1

window = 20
if len(xml_format) >= window:
    ma = np.convolve(xml_format, np.ones(window)/window, mode='valid')
    ma_steps = steps[window-1:len(xml_format)]
else:
    ma = []
    ma_steps = []

plt.figure(figsize=(10, 5))
plt.plot(steps, xml_format, label='XML Format Reward', alpha=0.5)
if len(ma_steps):
    plt.plot(ma_steps, ma, label=f'Moving Average (window={window})', color='red', linewidth=2)
plt.xlabel('Step')
plt.ylabel('XML Format Reward')
plt.title('XML Format Reward over Training Steps')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.25))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.tight_layout()
plt.show() 