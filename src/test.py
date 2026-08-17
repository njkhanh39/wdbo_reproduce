import numpy as np
from time import sleep, time
import gpytorch
from wdbo_algo.optimizer import WDBOOptimizer

# The expensive, noisy objective function
def my_noisy_objective_function(x, t):
	sleep(0.1)
	x = list(x) + [5.0 * t / 8.0 - 1.0]
	print(x)
	f = 0.0
	for i in range(2):
		f += 100 * ((x[i + 1] - x[i] ** 2) ** 2) + (1 - x[i]) ** 2

	return -f + np.random.normal(0.0, np.sqrt(0.1))

if __name__ == "__main__":
	# Spatial domain
	spatial_domain = np.array([[-1.0, 1.5] for _ in range(2)])

	# Optimizer instantiation
	optimizer = WDBOOptimizer(
		spatial_domain,
		gpytorch.kernels.MaternKernel, gpytorch.kernels.MaternKernel,
		spatial_kernel_args=[2.5], temporal_kernel_args=[2.5],
		n_initial_observations=15
	)

	# Start time
	start_time = 0.0
	# Experiment duration (minutes)
	xp_duration = 4.0
	# End time
	end_time = start_time + xp_duration
	# Optimization loop
	current_time = start_time

	# As long as the experiment is not over
	while current_time < end_time:
		start = time()
		# Find a relevant input to query
		x = optimizer.next_query(current_time)
		# Query the objective function
		y = my_noisy_objective_function(x, current_time)
		# Add the new observation to the dataset
		optimizer.tell(x, current_time, y)
		# Clean the dataset from irrelevant observations
		optimizer.clean(current_time)
		end = time()

		current_time += (end - start) / 60.0
