import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# The original Gaussian Mixture Model
# (The additional noise is bounded and of an order larger than the sample mean deviation)
means = [2, 10]
std_devs = [1, 1]

# Generation
total_generations = 100

# Sampling Generation
sampling_generations = 10

# Number Of Sample
samples_per_generation = 1000

# Original Data
original_data = np.concatenate(
    [np.random.normal(mean, std_dev, samples_per_generation // 2) for mean, std_dev in zip(means, std_devs)])

# Graph Setting
fig, axes = plt.subplots(total_generations // sampling_generations, 1, figsize=(5, 30), sharex=True)
plt.subplots_adjust(top=0.95, bottom=0.02, hspace=0.01)

# Plot distribution of data by generation
data = original_data.copy()
for generation in range(total_generations):
    if generation % sampling_generations == 0:
        ax = axes[generation // sampling_generations]
        ax.hist(original_data, bins=50, density=True, alpha=0.3, color='red', label='Original Data')
        ax.hist(data, bins=50, density=True, alpha=0.5, label=f'Generation {generation} Generated Data')
        ax.legend()

    # Fitting a Gaussian mixture model
    gmm = GaussianMixture(n_components=2)
    gmm.fit(data.reshape(-1, 1))

    # Sampling new data from the fitted model
    data = gmm.sample(samples_per_generation)[0].flatten()

    # Adding Noise
    noise = np.random.normal(0, 0.5, samples_per_generation)  # Adjust the standard deviation of noise
    data += noise

plt.xlabel('Value')
plt.ylabel('Density')
plt.suptitle('Model Collapse Simulation')
plt.show()




