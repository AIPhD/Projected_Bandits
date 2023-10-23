import numpy as np
import config as c
import real_data as rd
import training as t


def real_projected_training(target_context,
                            reward_data_complete,
                            proj_mat,
                            off_set,
                            estim,
                            dimension,
                            repeats=1,
                            decision_making='ucb',
                            exp_scale=1):
    '''Training algorithm based on linUCB and a biased regularization constrained.'''

    theta_estim = np.tile(estim, (repeats, 1))
    theta_estim_p = np.tile(estim, (repeats, 1))
    gamma_scalar = np.tile(c.GAMMA, (repeats, 1))
    rewards = np.zeros((repeats, c.EPOCHS))
    a_matrix_p = np.tile(c.LAMB_1 * np.identity(dimension) - (c.LAMB_1 - c.LAMB_2) * proj_mat[0],
                         (repeats, 1, 1))
    a_inv_p = np.tile(np.linalg.inv(c.LAMB_1 * np.identity(dimension) -
                                    (c.LAMB_1 - c.LAMB_2) * proj_mat[0]), (repeats, 1, 1))
    a_matrix = np.tile(c.LAMB_2 * np.identity(dimension), (repeats, 1, 1))
    a_inv = np.tile(1/c.LAMB_2 * np.identity(dimension), (repeats, 1, 1))
    inv_proj = np.tile(np.identity(dimension), (repeats, 1, 1)) - proj_mat

    b_vector_p = np.tile(np.zeros(dimension), (repeats, 1)) + c.LAMB_1 * np.einsum('ijk,ik->ij',
                                                                                     inv_proj,
                                                                                     off_set)
    b_vector = np.tile(np.zeros(dimension), (repeats, 1))
    regret_evol = np.zeros((repeats, c.EPOCHS))
    x_history = []
    index_list = []

    for i in range(0, c.EPOCHS):

        arm_ind = np.random.randint(len(target_context), size=c.ARM_SET)
        target_data = target_context# [arm_ind]
        reward_data = reward_data_complete# [arm_ind]

        if decision_making == 'ucb':
            index = np.argmax(t.ucb_function(target_data,
                                             a_inv_p,
                                             theta_estim_p,
                                             gamma_scalar,
                                             expl_scale=exp_scale), axis=1)
        elif decision_making == 'ts':
            index = np.argmax(t.ts_function(target_data,
                                            a_inv_p,
                                            theta_estim_p,
                                            exp_scale=exp_scale), axis=1)

        selected_item = np.random.randint(len(index))
        instance = target_data[[index[selected_item]]]
        index_list.append([index[selected_item]])
        x_history.append(instance)
        r_real = reward_data[[index[selected_item]]]
        rewards[:, i] = np.floor(r_real)
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
        gamma_scalar = np.asarray([1 +
                                   np.sqrt(np.log(np.linalg.det(a_matrix_p)/
                                                      (np.linalg.det(c.LAMB_2 * proj_mat +
                                                                     c.LAMB_1 * inv_proj) *
                                                                     c.DELTA**2)))]).T
        inst_regret = [np.max(reward_data)] - r_real
        regret_evol[:, i] = inst_regret
        # regr = inst_regret
        # print(f"Instant regret = {regr}")

    mean_regret = np.cumsum(regret_evol, axis=1).sum(axis=0)/repeats
    regret_dev = np.sqrt(np.sum((mean_regret - np.cumsum(regret_evol, axis=1))**2, axis=0)/repeats)



    return [mean_regret, regret_dev, theta_estim, a_matrix, b_vector, np.cumsum(regret_evol, axis=1)]


def real_meta_training(filtered_user_index,
                       context,
                       rewards,
                       dimension,
                       method='ccipca',
                       decision_making='ucb',
                       repeats=1,
                       high_bias=False,
                       exp_scale=1,
                       dim_known=False,
                       dim_set=None):
    '''Meta learning algorithm updating affine subspace after each training.'''
    theta_array = []
    j = 0
    learned_proj = np.tile(np.identity(dimension), (repeats, 1, 1))
    off_set = np.zeros((repeats, dimension))
    u_proj = None
    v_proj = None
    mean_regret_evol = np.zeros(c.EPOCHS)
    regret_evol = []
    regret_meta_rounds_evolution = np.zeros((repeats, len(filtered_user_index)))

    if high_bias:
        a_glob = np.zeros((repeats, dimension, dimension))
        a_inv_glob = np.zeros((repeats, dimension, dimension))
        b_glob = np.zeros((repeats, dimension))

    for i in np.arange(len(filtered_user_index)):
        real_target, real_reward = rd.extract_context_for_users(filtered_user_index[i],
                                                                context,
                                                                rewards)
        train_data = real_projected_training(real_target,
                                             real_reward,
                                             proj_mat=learned_proj,
                                             off_set=off_set,
                                             estim=np.zeros(dimension),
                                             dimension=dimension,
                                             exp_scale=exp_scale,
                                             decision_making=decision_making)
        regret_meta_rounds_evolution[:, i] = train_data[5][:, -1]

        # if train_data[0][-1] < 50:
        theta_array.append(train_data[2])
            # j += 1

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
                learned_proj, u_proj = t.online_pca(np.asarray(theta_array), u_proj)
            elif method == 'ccipca':
                learned_proj, v_proj = t.cc_ipca(np.asarray(theta_array),
                                                         None,
                                                         dim_known=dim_known,
                                                         dim_set=dim_set)
                inv_proj = np.tile(np.identity(dimension), (repeats, 1, 1)) - learned_proj
                off_set = np.einsum('ijk,ik->ij', inv_proj, theta_mean)
            elif method == 'full_dimension':
                learned_proj = np.zeros((repeats, dimension, dimension))
                inv_proj = np.tile(np.identity(dimension), (repeats, 1, 1))
                off_set = np.einsum('ijk,ik->ij', inv_proj, theta_mean)
            elif method == 'classic_learning':
                learned_proj = np.tile(np.identity(dimension), (repeats, 1, 1))
                inv_proj =  np.zeros((repeats, dimension, dimension))
                off_set = np.zeros((repeats, dimension))

            # if train_data[0][-1] < 50:
            j += 1
            regret_evol.append(train_data[0])
            mean_regret_evol += train_data[0]

    if j > 0:
        std_dev= np.sqrt(np.sum((mean_regret_evol/j - np.asarray(regret_evol))**2, axis=0)/j)
    else:
        std_dev= np.zeros(c.DIMENSION)

    mean_meta_regret = np.cumsum(regret_meta_rounds_evolution,
                                 axis=1).sum(axis=0)/len(regret_meta_rounds_evolution)
    regret_meta_dev = np.sqrt(np.sum((mean_meta_regret - np.cumsum(regret_meta_rounds_evolution, axis=1))**2,
                                     axis=0)/len(regret_meta_rounds_evolution))
    mean_proj = np.sum(learned_proj, axis=0)/(len(learned_proj))


    return [mean_regret_evol/j, std_dev, mean_meta_regret, regret_meta_dev]
