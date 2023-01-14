import numpy as np
import matplotlib.pyplot as plt
import config as c
import real_data as rd
import training as t

all_targets = rd.context_data_set
all_rewards = rd.reward_data_set


def real_projected_training(target_context,
                            reward_data_complete,
                            proj_mat=np.tile(np.identity(c.DIMENSION),(1,1,1)),
                            off_set=np.zeros((1, c.DIMENSION)),
                            estim=np.abs(np.random.uniform(size=c.DIMENSION)),
                            repeats=1,
                            decision_making='ucb',
                            arms_pulled_plot=False,
                            exp_scale=1):
    '''Training algorithm based on linUCB and a biased regularization constrained.'''

    theta_estim = estim/np.sqrt(np.dot(estim, estim))
    theta_estim = np.tile(theta_estim, (repeats, 1))
    theta_estim_p = np.tile(estim/np.sqrt(np.dot(estim, estim)), (repeats, 1))
    gamma_scalar = np.tile(c.GAMMA, (repeats, 1))
    no_pulls = np.zeros((repeats, c.CONTEXT_SIZE))
    rewards = np.zeros((repeats, c.EPOCHS))
    a_matrix_p = np.tile(c.LAMB_1 * np.identity(c.DIMENSION) - (c.LAMB_1 - c.LAMB_2) * proj_mat[0], (repeats, 1, 1))
    a_inv_p = np.tile(np.linalg.inv(c.LAMB_1 * np.identity(c.DIMENSION) -
                                    (c.LAMB_1 - c.LAMB_2) * proj_mat[0]), (repeats, 1, 1))
    a_matrix = np.tile(c.LAMB_2 * np.identity(c.DIMENSION), (repeats, 1, 1))
    a_inv = np.tile(1/c.LAMB_2 * np.identity(c.DIMENSION), (repeats, 1, 1))
    inv_proj = np.tile(np.identity(c.DIMENSION), (c.REPEATS, 1, 1)) - proj_mat

    b_vector_p = np.tile(np.zeros(c.DIMENSION), (repeats, 1)) + c.LAMB_1 * np.einsum('ijk,ik->ij',
                                                                                     inv_proj,
                                                                                     off_set)
    b_vector = np.tile(np.zeros(c.DIMENSION), (repeats, 1))
    regret_evol = np.zeros((repeats, c.EPOCHS))
    x_history = []
    index_list = []

    for i in range(0, c.EPOCHS):
        
        arm_ind = np.random.randint(c.CONTEXT_SIZE, size=c.ARM_SET)
        target_data = target_context[arm_ind]
        reward_data = reward_data_complete[arm_ind]

        if decision_making == 'ucb':
            index = np.argmax(t.ucb_function(target_data,
                                           a_inv_p,
                                           theta_estim_p,
                                           gamma_scalar,
                                           expl_scale=exp_scale), axis=1)
        elif decision_making == 'ts':
            index = np.argmax(t.ts_function(target_data,
                                          a_inv_p,
                                          theta_estim_p), axis=1)

        instance = target_data[index]
        index_list.append(index)
        x_history.append(instance)
        r_real = reward_data[index]
        rewards[:, i] = r_real
        # y_t = np.einsum('lij,ij->li', np.asarray(x_history), theta_estim).T
        a_matrix_p += np.einsum('ij,ik->ijk', instance, instance)
        a_matrix += np.einsum('ij,ik->ijk', instance, instance)
        a_inv_p -= np.einsum('ijk,ikl->ijl',
                             a_inv_p,
                             np.einsum('ijk,ikl->ijl',
                                       np.einsum('ij,ik->ijk',
                                                 instance,
                                                 instance),
                                       a_inv_p))/(1 + np.einsum('ij,ij->i',
                                                                instance,
                                                                np.einsum('ijk,ik->ij',
                                                                          a_inv_p,
                                                                          instance)))[:,
                                                                                      np.newaxis,
                                                                                      np.newaxis]
        a_inv -= np.einsum('ijk,ikl->ijl',
                           a_inv,
                           np.einsum('ijk,ikl->ijl',
                                     np.einsum('ij,ik->ijk',
                                               instance,
                                               instance),
                                     a_inv))/(1 + np.einsum('ij,ij->i',
                                                            instance,
                                                            np.einsum('ijk,ik->ij',
                                                                      a_inv,
                                                                      instance)))[:,
                                                                                  np.newaxis,
                                                                                  np.newaxis]
        b_vector_p += r_real[:, np.newaxis] * instance
        b_vector += r_real[:, np.newaxis] * instance
        theta_estim_p = np.einsum('ijk,ik->ij', a_inv_p, b_vector_p)
        theta_estim = np.einsum('ijk,ik->ij', a_inv, b_vector)
        # gamma_scalar = np.sqrt(np.max(np.abs(rewards[:, :i + 1] - y_t[:, :i + 1])/
        #                               np.sqrt(np.einsum('lij,lij->li',
        #                                                 np.asarray(x_history),
        #                                                 np.asarray(x_history))).T,
        #                               axis=1) + np.einsum('ij,ij->i',
        #                                                   y_t - rewards[:, :i+1],
        #                                                   y_t - rewards[:, :i+1]))[:, np.newaxis]
        gamma_scalar = np.asarray([np.sqrt(c.LAMB_2) + c.LAMB_1/np.sqrt(c.LAMB_2) * c.KAPPA +
                                   np.sqrt(1 * np.log(np.linalg.det(a_matrix_p)/
                                                      (c.LAMB_2**c.DIMENSION * c.DELTA**2)))]).T
        no_pulls[np.arange(repeats), index] += 1
        inst_regret = [np.max(reward_data)] - r_real
        regret_evol[:, i] = inst_regret
        # regr = inst_regret
        # print(f"Instant regret = {regr}")

    mean_regret = np.cumsum(regret_evol, axis=1).sum(axis=0)/repeats
    regret_dev = np.sqrt(np.sum((mean_regret - np.cumsum(regret_evol, axis=1))**2, axis=0)/repeats)

    if arms_pulled_plot:
        plt.scatter(reward_data,
                    np.sum(no_pulls, axis=0)/repeats)
        plt.show()
        plt.close()

    return [mean_regret, regret_dev, theta_estim, a_matrix, b_vector]


def real_meta_training(theta_opt_list,
                  target_context,
                  estim=np.abs(np.random.uniform(size=c.DIMENSION)),
                  method='ccipca',
                  decision_making='ucb',
                  repeats=1,
                  high_bias=False,
                  exp_scale=1,
                  dim_known=False):
    '''Meta learning algorithm updating affine subspace after each training.'''
    theta_array = []
    i = 0
    learned_proj = np.tile(np.identity(c.DIMENSION), (repeats, 1, 1))
    off_set = np.zeros((repeats, c.DIMENSION))
    u_proj = None
    v_proj = None

    if high_bias:
        a_glob = np.zeros((repeats, c.DIMENSION, c.DIMENSION))
        a_inv_glob = np.zeros((repeats, c.DIMENSION, c.DIMENSION))
        b_glob = np.zeros((repeats, c.DIMENSION))

    for theta_opt in theta_opt_list:
        train_data = real_projected_training(theta_opt,
                                        target_context,
                                        proj_mat=learned_proj,
                                        off_set=off_set,
                                        estim=estim,
                                        exp_scale=exp_scale,
                                        decision_making=decision_making)
        theta_array.append(train_data[2])

        if high_bias:
            a_glob += train_data[3]
            b_glob += train_data[4]

            for j in range(0, repeats):
                a_inv_glob[j] = np.linalg.inv(a_glob[j])

            theta_mean = np.einsum('ijk,ik->ij', a_inv_glob, b_glob)
        else:
            theta_mean = np.sum(np.asarray(theta_array), axis=0)/len(theta_array)

        if i > 50:
            if method == 'sga':
                learned_proj, u_proj = online_pca(np.asarray(theta_array), u_proj)
            elif method == 'ccipca':
                learned_proj, v_proj, u_proj = cc_ipca(np.asarray(theta_array),
                                                       v_proj,
                                                       u_proj,
                                                       dim_known=dim_known)
                inv_proj = np.tile(np.identity(c.DIMENSION), (repeats, 1, 1)) - learned_proj
                off_set = np.einsum('ijk,ik->ij', inv_proj, theta_mean)
            elif method == 'full_dimension':
                learned_proj = np.zeros((repeats, c.DIMENSION, c.DIMENSION))
                inv_proj = np.tile(np.identity(c.DIMENSION), (repeats, 1, 1))
                off_set = np.einsum('ijk,ik->ij', inv_proj, theta_mean)

        i += 1


    mean_proj = np.sum(learned_proj, axis=0)/len(learned_proj)

    return [train_data[0], train_data[1]]