import numpy as np

DIMENSION = 30
DIMENSION_ALIGN = 15
CONTEXT_SIZE = 1000
ARM_SET = 10
NO_TASK = 400
EPSILON = 1
SIGMA = 1 # 1/np.sqrt(2 * np.pi)
KAPPA = 0.1
V_MAX = 1
VAR_MAX = 10
LAMB_1 = 200     # 3.5          # Parameters used for real data run 20 for synthetic experiments
LAMB_2 = 10     # 0.1                                             1
EPOCHS = 250
DELTA = 0.1     # np.exp(-2 * LAMB_2)
LAMB_0 = 1/EPOCHS
REPEATS = 10
TASK_INIT = 50

try:
    GAMMA = SIGMA * np.sqrt(DIMENSION *
                            np.log(1 +
                                    1/(LAMB_2 * DIMENSION)) +
                            2 * np.log(1/DELTA)) + np.sqrt(LAMB_2)

except ZeroDivisionError:
    print("Lambda cannot be zero!")
    GAMMA = 0.1
