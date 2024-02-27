# The code was run on Python 3.8.10 in Ubuntu 20.04

Install numpy, scipy, matplotlib and sklearn

Go to evaluation.py, change user in PLOT_DIR string to the respective directory of this code's location

Within Projected_Bandits directory create two directories:

"data_sets" ; "plots"




# For synthetic experiments:

- go to run.py

- uncomment :     

	meta_learning_evaluation(task_params, arm_set, init_estim, proj_mat, off_set)

- run run.py

- Plots are located in Projected_Bandits/plots/real_data_experiments




# For real data experiments:

- Download Movielens data from https://grouplens.org/datasets/movielens/1m/

- copy paste: movies, ratings and users to Projected_Bandits/data_sets

- change LAMB_1, LAMB_2 in config.py to 3.5, 0.1 respectively

- Go to run.py and uncomment in the main funciton:

	    # real_data_comparison(gender='M', age=None, prof=None, max_tasks=400, y_top_limit= 40)

- run run.py

- Plots are located in Projected_Bandits/plots/synthetic_data_experiments
