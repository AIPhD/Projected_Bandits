import scipy
from scipy.stats import ortho_group
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import config as c
matplotlib.use('TkAgg')


def ucb_function(x_instances,
                 a_inv,
                 theta_t,
                 gamma_t,
                 expl_scale=1):
    '''Standard UCB function based on linear rewards.'''

    reward_estim_t = np.einsum('jk,ik->ij', x_instances, theta_t)
    expl = expl_scale * gamma_t * np.sqrt(np.einsum('jk,ikj->ij',
                                                    x_instances,
                                                    np.einsum('ijk,lk->ijl',
                                                              a_inv,
                                                              x_instances)))
    return reward_estim_t + expl

def ts_function(x_instances,
                a_inv,
                theta_t):
    '''Function handling the thompson sampling with shared subspaces.'''
    epsilon = 0.99
    uncertainty_scale = 0.00002 * c.SIGMA**2 * 96/epsilon * len(a_inv[0]) * np.log(1/c.DELTA)
    theta_tild = np.zeros((len(a_inv), len(theta_t[0])))

    for i in np.arange(len(a_inv)):
        cov_a = uncertainty_scale * a_inv[i]
        if np.isnan(np.min(np.abs(cov_a))) or np.isinf(np.max(np.abs(cov_a))):
            print('Nan value in covariance')
        theta_tild[i] = np.random.multivariate_normal(theta_t[i], cov_a)

    return np.einsum('jk,ik->ij', x_instances, theta_tild)


def online_pca(theta_data, u_proj=None, learning_rate=1, momentum_scale=0.99):
    '''Online PCA function in order to determine affine subspace.'''
    if u_proj is None:
        u_proj = ortho_group.rvs(dim=c.DIMENSION)
        u_proj = np.delete(u_proj, np.arange(c.DIMENSION_ALIGN), axis=1)
        u_proj = np.tile(u_proj, (c.REPEATS, 1, 1))
    mean_theta = np.sum(theta_data, axis=0)/len(theta_data)
    gamma_theta = 1/len(theta_data) * np.sum(np.einsum('mik,mil->mikl',
                                                       theta_data - mean_theta,
                                                       theta_data - mean_theta), axis=0)

    for k in np.arange(c.REPEATS):
        step_scale = 1
        while np.linalg.norm(np.einsum('kl,lj->kj',
                                       gamma_theta[k],
                                       u_proj[k])) > 0.01 and step_scale < 15000:
            sga_step = 0
            sga_step = learning_rate/step_scale * (np.einsum('kl,lj->kj',
                                                             gamma_theta[k],
                                                             u_proj[k]) +
                                                   momentum_scale * sga_step)
            u_proj[k] += sga_step
            step_scale += 1

        print(step_scale)

    for i in range(0, c.REPEATS):
        u_proj[i] = scipy.linalg.orth(u_proj[i]) #.dot(inv(sqrtm(u_proj[i].T.dot(u_proj[i]))))

    proj_mat = np.einsum('ikj,ilj->ikl', u_proj, u_proj)
    return proj_mat, u_proj


def cc_ipca(theta_data, v_proj=None, u_proj=None, dim_known=False, dim_set=c.DIMENSION_ALIGN):
    '''Covariance free strategy to solve online pca, without relying on hyper parameters.'''
    dimension = len(theta_data[0][0])
    repeats = len(theta_data[0])
    dim_align_counter = np.zeros(repeats)
    len_t = len(theta_data)
    mean_theta = np.sum(theta_data, axis=0)/len_t
    gamma_theta = 1/len_t * np.sum(np.einsum('mik,mil->mikl',
                                             theta_data - mean_theta,
                                             theta_data - mean_theta), axis=0)
    if v_proj is None:
        eig_v, u_proj = np.linalg.eigh(gamma_theta)
        v_proj = np.zeros((repeats, dimension, dimension))

        for i in np.arange(len(u_proj)):
            v_proj[i] = eig_v[i] * u_proj[i]

    else:
        # x_proj = np.zeros((repeats, dimension, dimension))
        # prod_x = np.zeros((repeats, dimension, dimension))

        # for j in np.arange(repeats):
        #     prod_x[j] = np.einsum('ik,ikm->im',
        #                           theta_data[-1, :, :],
        #                           u_proj)[j] * u_proj[j]

        # for k in np.arange(len(u_proj[0])):
        #     x_proj[:, :, k] = theta_data[-1] - np.sum(prod_x[:, :, :k+1],
        #                                               axis=2)

        # v_proj = len_t/(len_t + 1) * v_proj + 1/(len_t + 1) * np.einsum('iklj,ilj->ikj',
        #                                                                 np.einsum('ikj,ilj->iklj',
        #                                                                           x_proj,
        #                                                                           x_proj),
        #                                                                 u_proj)


        x_proj = np.zeros((repeats, dimension))
        prod_x = np.zeros((repeats, dimension))


        prod_x = np.einsum('im, ikm-> ik',
                           np.einsum('ik,ikm->im',
                                     theta_data[-1, :, :],
                                     u_proj),
                           u_proj)

        x_proj = theta_data[-1] - prod_x

        v_proj = len_t/(len_t + 1) * v_proj + 1/(len_t + 1) * np.einsum('ikl,ilj->ikj',
                                                                        np.einsum('ik,il->ikl',
                                                                                  x_proj,
                                                                                  x_proj),
                                                                        u_proj)

        eig_v = np.linalg.norm(v_proj, axis=1)

    arg_sorted_eig = np.argsort(eig_v, axis=1)


    for i in np.arange(len(dim_align_counter)):
        for j in np.arange(len(eig_v[0])):

            if eig_v[i][j] < 0.1:
                dim_align_counter[i] += 1

    # for i in np.arange(len(dim_align_counter)):
    #     dim_align_counter[i] = np.argmax(np.diff(eig_v, axis=1, prepend=0)[0])


    for i in np.arange(repeats):

        if dim_known:
            shared_dim = dim_set
        else:
            shared_dim = dim_align_counter[i]

        u_proj[i] = normalize(v_proj[i], axis=0, norm='l2')

        for j in np.arange(shared_dim):
            u_proj[i][:, np.where(arg_sorted_eig[i]==j)[0]]=np.zeros(dimension)[np.newaxis].T

    proj_mat = np.einsum('ikj,ilj->ikl', u_proj, u_proj)
    print(dim_align_counter)
    return proj_mat, v_proj, u_proj


def projected_training(theta_opt,
                       target_context,
                       proj_mat=np.tile(np.identity(c.DIMENSION),(c.REPEATS,1,1)),
                       off_set=np.zeros((c.REPEATS, c.DIMENSION)),
                       estim=np.zeros(c.DIMENSION),
                       repeats=c.REPEATS,
                       decision_making='ucb',
                       arms_pulled_plot=False,
                       exp_scale=1,
                       meta_theta=None,
                       meta_cov=None):
    '''Training algorithm based on linUCB and a biased regularization constrained.'''

    theta_estim_p = np.tile(estim, (repeats, 1))
    theta_target = np.tile(theta_opt, (repeats, 1))
    gamma_scalar = np.tile(c.GAMMA, (repeats, 1))
    no_pulls = np.zeros((repeats, c.CONTEXT_SIZE))
    rewards = np.zeros((repeats, c.EPOCHS))
    a_matrix_p = np.zeros((repeats, c.DIMENSION, c.DIMENSION))
    a_inv_p = np.zeros((repeats, c.DIMENSION, c.DIMENSION))
    a_matrix = np.tile(c.LAMB_2 * np.identity(c.DIMENSION), (repeats, 1, 1))
    a_inv = np.tile(1/c.LAMB_2 * np.identity(c.DIMENSION), (repeats, 1, 1))
    inv_proj = np.tile(np.identity(c.DIMENSION), (c.REPEATS, 1, 1)) - proj_mat
    if decision_making == 'meta_prior':
        if meta_cov is None:
            prior_mean = np.zeros((repeats, c.DIMENSION))
            mean_a = np.zeros((repeats, c.DIMENSION))
            prior_cov_inv = a_matrix
            cov_a = a_inv
        else:
            prior_mean = meta_theta
            mean_a = meta_theta
            cov_a = meta_cov
            prior_cov_inv = np.zeros((repeats, c.DIMENSION, c.DIMENSION))
            for i in np.arange(repeats):
                prior_cov_inv[i] = np.linalg.inv(meta_cov[i])
    for i in range(0, c.REPEATS):
        a_matrix_p[i] = c.LAMB_1 * np.identity(c.DIMENSION) - (c.LAMB_1 - c.LAMB_2) * proj_mat[i]
        a_inv_p[i] = np.linalg.inv(c.LAMB_1 * np.identity(c.DIMENSION) -
                                 (c.LAMB_1 - c.LAMB_2) * proj_mat[i])

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

        if decision_making == 'meta_prior':
            if i < c.DIMENSION + 10:
                index = np.random.randint(len(target_data), size=repeats)
            else:
                index = np.argmax(ts_function(target_data,
                                              cov_a,
                                              mean_a), axis=1)

        if decision_making == 'ucb':
            index = np.argmax(ucb_function(target_data,
                                           a_inv_p,
                                           theta_estim_p,
                                           gamma_scalar,
                                           expl_scale=exp_scale), axis=1)
        elif decision_making == 'ts':
            if np.isnan(np.min(np.abs(a_inv_p))) or np.isinf(np.max(np.abs(a_inv_p))):
                print('Nan value in covariance')
            index = np.argmax(ts_function(target_data,
                                          a_inv_p,
                                          theta_estim_p), axis=1)

        index_opt = np.argmax(np.einsum('ij,kj->ik',theta_target, target_data), axis=1)
        instance = target_data[index]
        index_list.append(index)
        x_history.append(instance)
        opt_instance = target_data[index_opt]
        noise = c.EPSILON * np.random.normal(scale=c.SIGMA, size=repeats)
        r_real = np.einsum('ij,ij->i', theta_target, instance) + noise
        rewards[:, i] = r_real
        # y_t = np.einsum('lij,ij->li', np.asarray(x_history), theta_estim).T
        a_matrix_p += np.einsum('ij,ik->ijk', instance, instance)
        a_matrix += np.einsum('ij,ik->ijk', instance, instance)
        # for i in np.arange(repeats):
        #     a_inv_p[i] = np.linalg.inv(a_matrix_p[i])
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
        if np.isnan(np.min(np.abs(a_inv_p))) or np.isinf(np.max(np.abs(a_inv_p))):
            print('Nan value in covariance')
        b_vector_p += r_real[:, np.newaxis] * instance
        b_vector += r_real[:, np.newaxis] * instance
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

        if decision_making=='meta_prior':
            cov_a -= np.einsum('ijk,ikl->ijl',
                            cov_a,
                            np.einsum('ijk,ikl->ijl',
                                        np.einsum('ij,ik->ijk',
                                                1/c.SIGMA*instance,
                                                1/c.SIGMA*instance),
                                        cov_a))/(1 + np.einsum('ij,ij->i',
                                                                1/c.SIGMA*instance,
                                                                np.einsum('ijk,ik->ij',
                                                                        cov_a,
                                                                        1/c.SIGMA*instance)))[:,
                                                                                    np.newaxis,
                                                                                    np.newaxis]
            mean_a = np.einsum('ijk,ik->ij',
                               cov_a,
                               np.einsum('ijk,ik->ij', prior_cov_inv, prior_mean) + b_vector)

        theta_estim_p = np.einsum('ijk,ik->ij', a_inv_p, b_vector_p)
        # gamma_scalar = np.sqrt(np.max(np.abs(rewards[:, :i + 1] - y_t[:, :i + 1])/
        #                               np.sqrt(np.einsum('lij,lij->li',
        #                                                 np.asarray(x_history),
        #                                                 np.asarray(x_history))).T,
        #                               axis=1) + np.einsum('ij,ij->i',
        #                                                   y_t - rewards[:, :i+1],
        #                                                   y_t - rewards[:, :i+1]))[:, np.newaxis]
        gamma_scalar = np.asarray([np.sqrt(c.LAMB_2) + c.LAMB_1/np.sqrt(c.LAMB_2) * c.KAPPA +
                                   np.sqrt(1 * np.log(np.linalg.det(a_matrix_p)/
                                                      (np.linalg.det(c.LAMB_2 * proj_mat +
                                                                     c.LAMB_1 * inv_proj) *
                                                                     c.DELTA**2)))]).T
        no_pulls[np.arange(repeats), index] += 1
        inst_regret = np.einsum('ij,ij->i', theta_target, opt_instance) - np.einsum('ij,ij->i',
                                                                                    theta_target,
                                                                                    instance)
        regret_evol[:, i] = inst_regret
        # regr = inst_regret
        # print(f"Instant regret = {regr}")

    if decision_making == "meta_prior":
        for i in np.arange(repeats):
            a_inv[i] = np.linalg.inv(np.matmul(np.asarray(x_history)[:, i, :].T,
                                               np.asarray(x_history)[:, i, :]))
        theta_estim = np.einsum('ijk,ik->ij', a_inv, b_vector)
    else:
        theta_estim = np.einsum('ijk,ik->ij', a_inv, b_vector)

    mean_regret = np.cumsum(regret_evol, axis=1).sum(axis=0)/repeats
    regret_dev = np.sqrt(np.sum((mean_regret - np.cumsum(regret_evol, axis=1))**2, axis=0)/repeats)

    if arms_pulled_plot:
        plt.scatter(np.sum(np.einsum('ijk,ik->ij',
                                     target_data,
                                     theta_target),
                           axis=0)/repeats,
                    np.sum(no_pulls, axis=0)/repeats)
        plt.show()
        plt.close()

    return [mean_regret, regret_dev, theta_estim, a_matrix, b_vector, a_inv, np.cumsum(regret_evol, axis=1)]


def meta_training(theta_opt_list,
                  target_context,
                  estim=np.abs(np.random.uniform(size=c.DIMENSION)),
                  method='ccipca',
                  decision_making='ucb',
                  high_bias=False,
                  exp_scale=1,
                  ideal_proj=None,
                  ideal_offset=None,
                  dim_known=False,
                  dim_set=c.DIMENSION_ALIGN):
    '''Meta learning algorithm updating affine subspace after each training.'''
    theta_array = []
    a_inv_array = []
    i = 0
    if ideal_proj is None and ideal_offset is None:
        learned_proj = np.tile(np.identity(c.DIMENSION), (c.REPEATS, 1, 1))
        off_set = np.zeros((c.REPEATS, c.DIMENSION))
    else:
        learned_proj = ideal_proj
        off_set = ideal_offset

    u_proj = None
    v_proj = None
    mean_regret_evol = np.zeros(c.EPOCHS)
    regret_evol = []
    regret_meta_rounds_evolution = np.zeros((c.REPEATS, c.NO_TASK))
    meta_theta = None
    meta_cov = None

    if high_bias:
        a_glob = np.zeros((c.REPEATS, c.DIMENSION, c.DIMENSION))
        a_inv_glob = np.zeros((c.REPEATS, c.DIMENSION, c.DIMENSION))
        b_glob = np.zeros((c.REPEATS, c.DIMENSION))

    for theta_opt in theta_opt_list:
        train_data = projected_training(theta_opt,
                                        target_context,
                                        proj_mat=learned_proj,
                                        off_set=off_set,
                                        estim=estim,
                                        exp_scale=exp_scale,
                                        decision_making=decision_making,
                                        meta_theta=meta_theta,
                                        meta_cov=meta_cov)
        regret_meta_rounds_evolution[:, i] = train_data[6][:, -1]
        theta_array.append(train_data[2])
        a_inv_array.append(train_data[5])

        if high_bias:
            a_glob += train_data[3]
            b_glob += train_data[4]

            for j in range(0, c.REPEATS):
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
                                                       dim_known=dim_known,
                                                       dim_set=dim_set)
                inv_proj = np.tile(np.identity(c.DIMENSION), (c.REPEATS, 1, 1)) - learned_proj
                off_set = np.einsum('ijk,ik->ij', inv_proj, theta_mean)
            elif method == 'full_dimension':
                learned_proj = np.zeros((c.REPEATS, c.DIMENSION, c.DIMENSION))
                inv_proj = np.tile(np.identity(c.DIMENSION), (c.REPEATS, 1, 1))
                off_set = np.einsum('ijk,ik->ij', inv_proj, theta_mean)
            elif method == 'classic_learning':
                learned_proj = np.tile(np.identity(c.DIMENSION), (c.REPEATS, 1, 1))
                inv_proj =  np.zeros((c.REPEATS, c.DIMENSION, c.DIMENSION))
                off_set = np.zeros((c.REPEATS, c.DIMENSION))
            elif method == 'ideal_proj':
                inv_proj = np.tile(np.identity(c.DIMENSION), (c.REPEATS, 1, 1)) - learned_proj
            elif method == 'meta_prior':
                learned_proj = np.tile(np.identity(c.DIMENSION), (c.REPEATS, 1, 1))
                inv_proj =  np.zeros((c.REPEATS, c.DIMENSION, c.DIMENSION))
                off_set = np.zeros((c.REPEATS, c.DIMENSION))
                meta_theta = theta_mean
                g_sigma = 1/(i-1) * np.sum(np.asarray(a_inv_array), axis=0)
                sigma_n = 1/(i-2) * np.sum(np.einsum('tij,tik->tijk',
                                                     np.asarray(theta_array) - meta_theta,
                                                     np.asarray(theta_array) - meta_theta),
                                           axis=0) - g_sigma
                c_w = 50 * (2/(c.DIMENSION)+ 1)
                meta_cov = sigma_n + c_w*np.sqrt((5*c.DIMENSION +
                                                  2*np.log(c.DIMENSION*
                                                           i*c.EPOCHS))/(i - 1))*np.tile(np.identity(c.DIMENSION),
                                                                                         (c.REPEATS, 1, 1))


            regret_evol.append(train_data[0])
            mean_regret_evol += train_data[0]

        i += 1


    std_dev = np.sqrt(np.sum((mean_regret_evol/(i-50) - np.asarray(regret_evol))**2, axis=0)/(i-50))
    mean_meta_regret = np.cumsum(regret_meta_rounds_evolution,
                                 axis=1).sum(axis=0)/len(regret_meta_rounds_evolution)
    regret_meta_dev = np.sqrt(np.sum((mean_meta_regret - np.cumsum(regret_meta_rounds_evolution,
                                                                   axis=1))**2,
                                     axis=0)/len(regret_meta_rounds_evolution))

    mean_proj = np.sum(learned_proj, axis=0)/len(learned_proj)

    # return [train_data[0], train_data[1]]
    return [mean_regret_evol/(i-50), std_dev, mean_meta_regret, regret_meta_dev]
