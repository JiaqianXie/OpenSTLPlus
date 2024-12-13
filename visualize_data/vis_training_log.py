import re
import matplotlib.pyplot as plt

# Path to the log file
log_file_path = './20241210_220528.log'

# Data storage
epochs = []
precision_event = []
precision_noise = []
recall_event = []
recall_noise = []
f1_event = []
f1_noise = []

# Pattern to extract the required data
pattern = re.compile(
    r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\].*?"
    r"multi-label/precision_top1_classwise: \[([\d\.]+), ([\d\.]+)\].*?"
    r"multi-label/recall_top1_classwise: \[([\d\.]+), ([\d\.]+)\].*?"
    r"multi-label/f1-score_top1_classwise: \[([\d\.]+), ([\d\.]+)\]"
)

# Read the log file and extract data
with open(log_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            precision_event.append(float(match.group(2)))
            precision_noise.append(float(match.group(3)))
            recall_event.append(float(match.group(4)))
            recall_noise.append(float(match.group(5)))
            f1_event.append(float(match.group(6)))
            f1_noise.append(float(match.group(7)))

# Plotting
def plot_metric(epochs, event_values, noise_values, metric_name, class_name):
    plt.figure(figsize=(10, 6))
    if class_name == 'event':
        plt.plot(epochs, event_values, label=f'{metric_name} - Event', marker='o')
    else:
        plt.plot(epochs, noise_values, label=f'{metric_name} - Noise', marker='x')
    plt.title(f'{metric_name} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{class_name}_{metric_name}.png')

# Plot each metric
plot_metric(epochs, precision_event, precision_noise, 'Precision@1', class_name="event")
plot_metric(epochs, recall_event, recall_noise, 'Recall@1', class_name="event")
plot_metric(epochs, f1_event, f1_noise, 'F1-Score@1', class_name="event")
plot_metric(epochs, precision_event, precision_noise, 'Precision@1', class_name="noise")
plot_metric(epochs, recall_event, recall_noise, 'Recall@1', class_name="noise")
plot_metric(epochs, f1_event, f1_noise, 'F1-Score@1', class_name="noise")
