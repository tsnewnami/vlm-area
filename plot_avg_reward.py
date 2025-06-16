import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

def extract_avg_rewards(log_file):
    avg_rewards = []
    pattern = re.compile(r"rewards_all: tensor\(\[([^\]]+)\]")
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Extract numbers, remove spaces, split by comma
                numbers = [float(x) for x in match.group(1).replace(' ', '').split(',') if x]
                if numbers:
                    avg = np.mean(numbers)
                    avg_rewards.append(avg)
    return avg_rewards

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main(log_file, window_size):
    avg_rewards = extract_avg_rewards(log_file)
    avg_rewards = np.array(avg_rewards)
    ma = moving_average(avg_rewards, window_size)
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards, label='Average Reward per Iteration', alpha=0.3, linewidth=1)
    plt.plot(range(window_size-1, len(avg_rewards)), ma, label=f'Moving Average (window={window_size})', color='red', linewidth=2)
    plt.xlabel('Training Iteration')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_reward.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot average reward from log file.")
    parser.add_argument('--log_file', type=str, default='log-good.log', help='Path to the log file')
    parser.add_argument('--window', type=int, default=20, help='Moving average window size')
    args = parser.parse_args()
    main(args.log_file, args.window) 