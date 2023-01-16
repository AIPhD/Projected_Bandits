import numpy as np

DIMENSION = 30
DIMENSION_ALIGN = 12
CONTEXT_SIZE = 1000
ARM_SET = 19
NO_TASK = 60
EPSILON = 1
SIGMA = 1/np.sqrt(2 * np.pi)
KAPPA = 0.1
LAMB_1 = 10
LAMB_2 = 1
DELTA = 0.1 # np.exp(-2 * LAMB_2)
EPOCHS = 250
REPEATS = 10

try:
    GAMMA = SIGMA * np.sqrt(DIMENSION *
                            np.log(1 +
                                    1/(LAMB_2 * DIMENSION)) +
                            2 * np.log(1/DELTA)) + np.sqrt(LAMB_2)

except ZeroDivisionError:
    print("Lambda cannot be zero!")
    GAMMA = 0.1
