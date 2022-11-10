import scipy
from scipy.stats import ortho_group
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import config as c
matplotlib.use('TKAgg')


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
                                       u_proj[k])) > 0.02 and step_scale < 15000:
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


def cc_ipca(theta_data, v_proj=None, u_proj=None):
    '''Covariance free strategy to solve online pca, without relying on hyper parameters.'''
    len_t = len(theta_data)
    mean_theta = np.sum(theta_data, axis=0)/len_t
    gamma_theta = 1/len_t * np.sum(np.einsum('mik,mil->mikl',
                                             theta_data - mean_theta,
                                             theta_data - mean_theta), axis=0)
    if v_proj is None:
        eig_v, u_proj = np.linalg.eigh(gamma_theta)
        v_proj = np.zeros((c.REPEATS, c.DIMENSION, c.DIMENSION))
        for i in np.arange(len(u_proj)):
            v_proj[i] = eig_v[i] * u_proj[i]
    else:
        x_proj = np.zeros((c.REPEATS, c.DIMENSION, c.DIMENSION))
        prod_x = np.zeros((c.REPEATS, c.DIMENSION, c.DIMENSION))

        for j in np.arange(c.REPEATS):
            prod_x[j] = np.einsum('ik,ikm->im',
                                  theta_data[-1, :, :],
                                  u_proj)[j] * u_proj[j]

        for k in np.arange(len(u_proj)):
            x_proj[:, :, k] = theta_data[-1] - np.sum(prod_x[:, :, :k+1],
                                                      axis=2)

        v_proj = len_t/(len_t + 1) * v_proj + 1/(len_t + 1) * np.einsum('iklj,ilj->ikj',
                                                                        np.einsum('ikj,ilj->iklj',
                                                                                  x_proj,
                                                                                  x_proj),
                                                                        u_proj)
        eig_v = np.linalg.norm(v_proj, axis=1)
    arg_sorted_eig = np.argsort(eig_v, axis=1)
    for i in np.arange(c.REPEATS):
        u_proj[i] = normalize(v_proj[i], axis=0, norm='l2')
        for j in np.arange(c.DIMENSION_ALIGN):
            u_proj[i][:, np.where(arg_sorted_eig[i]==j)[0]]=np.zeros(c.DIMENSION)[np.newaxis].T

    proj_mat = np.einsum('ikj,ilj->ikl', u_proj, u_proj)
    return proj_mat, v_proj, u_proj


def projected_training(theta_opt,
                       target_context,
                       proj_mat=np.tile(np.identity(c.DIMENSION),(c.REPEATS,1,1)),
                       estim=np.abs(np.random.uniform(size=c.DIMENSION)),
                       repeats=c.REPEATS,
                       arms_pulled_plot=False,
                       exp_scale=1):
    '''Training algorithm based on linUCB and a biased regularization constrained.'''

    theta_estim = estim/np.sqrt(np.dot(estim, estim))
    theta_estim = np.tile(theta_estim, (repeats, 1))

    target_data = target_context
    theta_target = np.tile(theta_opt, (repeats, 1))
    gamma_scalar = np.tile(c.GAMMA, (repeats, 1))
    no_pulls = np.zeros((repeats, c.CONTEXT_SIZE))
    rewards = np.zeros((repeats, c.EPOCHS))
    a_matrix = np.zeros((repeats, c.DIMENSION, c.DIMENSION))
    a_inv = np.zeros((repeats, c.DIMENSION, c.DIMENSION))

    for i in range(0, c.REPEATS):
        a_matrix[i] = c.LAMB_1 * np.identity(c.DIMENSION) - (c.LAMB_1 - c.LAMB_2) * proj_mat[i]
        a_inv[i] = np.linalg.inv(c.LAMB_1 * np.identity(c.DIMENSION) -
                                 (c.LAMB_1 - c.LAMB_2) * proj_mat[i])


    b_vector = np.tile(np.zeros(c.DIMENSION), (repeats, 1))
    regret_evol = np.zeros((repeats, c.EPOCHS))
    x_history = []
    index_list = []

    for i in range(0, c.EPOCHS):

        index = np.argmax(ucb_function(target_data,
                                       a_inv,
                                       theta_estim,
                                       gamma_scalar,
                                       expl_scale=exp_scale), axis=1)

        index_opt = np.argmax(np.einsum('ij,kj->ik',theta_target, target_data), axis=1)
        instance = target_data[index]
        index_list.append(index)
        x_history.append(instance)
        opt_instance = target_data[index_opt]
        noise = c.EPSILON * np.random.normal(scale=c.SIGMA, size=repeats)
        r_real = np.einsum('ij,ij->i', theta_target, instance) + noise
        rewards[:, i] = r_real
        # y_t = np.einsum('lij,ij->li', np.asarray(x_history), theta_estim).T
        a_matrix += np.einsum('ij,ik->ijk', instance, instance)
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
        b_vector += r_real[:, np.newaxis] * instance
        theta_estim = np.einsum('ijk,ik->ij', a_inv, b_vector)
        # gamma_scalar = np.sqrt(np.max(np.abs(rewards[:, :i + 1] - y_t[:, :i + 1])/
        #                               np.sqrt(np.einsum('lij,lij->li',
        #                                                 np.asarray(x_history),
        #                                                 np.asarray(x_history))).T,
        #                               axis=1) + np.einsum('ij,ij->i',
        #                                                   y_t - rewards[:, :i+1],
        #                                                   y_t - rewards[:, :i+1]))[:, np.newaxis]
        gamma_scalar = np.asarray([np.sqrt(c.LAMB_2) + c.LAMB_1/np.sqrt(c.LAMB_2) * c.KAPPA +
                                   np.sqrt(1 * np.log(np.linalg.det(a_matrix)/
                                                      (c.LAMB_2**c.DIMENSION * c.DELTA**2)))]).T
        no_pulls[np.arange(repeats), index] += 1
        inst_regret = np.einsum('ij,ij->i', theta_target, opt_instance) - np.einsum('ij,ij->i',
                                                                                    theta_target,
                                                                                    instance)
        regret_evol[:, i] = inst_regret
        # regr = inst_regret
        # print(f"Instant regret = {regr}")

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

    return mean_regret, regret_dev, theta_estim


def meta_training(theta_opt_list,
                  target_context,
                  estim=np.abs(np.random.uniform(size=c.DIMENSION)),
                  method='ccipca',
                  exp_scale=1):
    '''Meta learning algorithm updating affine subspace after each training.'''
    theta_array = []
    i = 0
    learned_proj = np.tile(np.identity(c.DIMENSION), (c.REPEATS, 1, 1))
    u_proj = None
    v_proj = None

    for theta_opt in theta_opt_list:
        regret, regret_dev, theta = projected_training(theta_opt,
                                                       target_context,
                                                       proj_mat=learned_proj,
                                                       estim=estim,
                                                       exp_scale=exp_scale)
        theta_array.append(theta)

        if i > 50:
            if method == 'sga':
                learned_proj, u_proj = online_pca(np.asarray(theta_array), u_proj)
            elif method == 'ccipca':
                learned_proj, v_proj, u_proj = cc_ipca(np.asarray(theta_array), v_proj, u_proj)

        i += 1

    mean_proj = np.sum(learned_proj, axis=0)/len(learned_proj)

    print(mean_proj)

    return regret, regret_dev