import numpy as np
import synthetic_data as sd
import config as c
import training as t
import evaluation as e
import real_data as rd
import real_training as rt


def main():
    '''Main function used to evaluate different bandit settings.'''

    multi_tasks = sd.TaskData()
    arm_set = multi_tasks.target_context
    task_params = multi_tasks.theta_opt
    proj_mat = multi_tasks.subspace_projection
    off_set = np.tile(multi_tasks.off_set, (c.REPEATS, 1))
    proj_mat = np.tile(proj_mat, (c.REPEATS, 1, 1))
    init_estim = np.zeros(c.DIMENSION) # np.abs(np.random.uniform(size=c.DIMENSION))

    filter_data = [
                #    ['F', 18, '0', True],
                #    ['F', 18, '0', False],
                #    ['F', 35, '12', True],
                #    ['F', 35, '12', False],
                #    ['F', 25, '4', False],
                #    ['F', 25, '4', True],
                #    ['F', 35, '6', False],
                #    ['F', 35, '6', True],
                #    ['F', 35, '9', False],
                #    ['F', 35, '9', True],
                   ['F', 35, '11', True],
                #    ['F', 35, '11', False],
                #    ['F', 35, '19', False],
                #    ['F', 35, '19', True],
                #    ['F', 45, '15', True],
                #    ['F', 45, '15', False],
                #    ['M', 18, '0', True],
                #    ['M', 18, '0', False],
                #    ['M', 35, '2', False],
                #    ['M', 35, '2', True],
                #    ['M', 35, '3', False],
                #    ['M', 35, '3', True],
                   ['M', 25, '12', False],
                   ['M', 35, '11', False],
                   ['M', 45, '15', True],
                #    ['M', 35, '17', False],
                #    ['M', 35, '17', True],
                #   ['M', 45, '3', True],
                #    ['M', 45, '3', False],
                #    ['M', 45, '20', False],
                #    ['M', 45, '20', True],
                #    ['M', 56, '13', True],
                #    ['M', 56, '13', False]
                   ]

    # for filt in filter_data:
    #     print(filt[0]+str(filt[1]))
    #     real_data_comparison(gender=filt[0], age=filt[1], prof=filt[2])

    # for i in np.arange(21):
    #     real_data_comparison(gender=None, age=None, prof=str(i))
    meta_learning_evaluation(task_params, arm_set, init_estim, proj_mat, off_set)
    # multi_task_evaluation(task_params, arm_set, init_estim, proj_mat)


def real_data_comparison(gender=None,
                         age=None,
                         prof=None,
                         context=rd.context_data_set,
                         rewards=rd.reward_data_set):
    '''Compares multiple algorithms on real data'''

    dimension = len(context[0])
    explore_scale = 0.1
    filtered_users = rd.filter_users(np.asarray(rd.user_data_set),
                                     gender=gender,
                                     age=age,
                                     prof=prof)
    filtered_user_index = np.asarray([int(i) for i in filtered_users[:, 0]]) - 1
    # real_target, real_rewards = rd.extract_context_for_users(filtered_user_index[-1],
    #                                                          context,
    #                                                          rewards)

    ts_data = rt.real_meta_training(filtered_user_index,
                                    context,
                                    rewards,
                                    dimension,
                                    method='ccipca',
                                    decision_making='ts',
                                    exp_scale=explore_scale)
    ipca_dimunknown_data = rt.real_meta_training(filtered_user_index,
                                                 context,
                                                 rewards,
                                                 dimension,
                                                 method='ccipca',
                                                 exp_scale=explore_scale)
    cella_data = rt.real_meta_training(filtered_user_index,
                                       context,
                                       rewards,
                                       dimension,
                                       method='full_dimension',
                                       exp_scale=explore_scale)
    linucb_data = rt.real_meta_training(filtered_user_index,
                                        context,
                                        rewards,
                                        dimension,
                                        method='classic_learning',
                                        exp_scale=explore_scale)
    thomps_data = rt.real_meta_training(filtered_user_index,
                                        context,
                                        rewards,
                                        dimension,
                                        method="classic_learning",
                                        exp_scale=explore_scale,
                                        decision_making='ts')

    e.multiple_regret_plots([linucb_data[0]],
                            [linucb_data[1]],
                            plot_label="LinUCB")
    e.multiple_regret_plots([thomps_data[0]],
                            [thomps_data[1]],
                            plot_label="Thompson Sampling")
    e.multiple_regret_plots([cella_data[0]],
                            [cella_data[1]],
                            plot_label="Cella et al., 2020")
    e.multiple_regret_plots([ts_data[0]],
                            [ts_data[1]],
                            plot_label="Projected Thompson Sampling")
    e.multiple_regret_plots([ipca_dimunknown_data[0]],
                            [ipca_dimunknown_data[1]],
                            plot_label="Projected LinUCB",
                            directory="real_data_experiments",
                            plotsuffix=f'real_regret_comparison_{gender}_{age}_{prof}',
                            do_plot=True)


def meta_learning_evaluation(task_params, arm_set, init_estim, real_proj, off_set):
    '''Evaluation of the meta learning task.'''

    # high_varianca_data = t.meta_training(task_params,
    #                                      arm_set,
    #                                      estim=init_estim,
    #                                      method='ccipca',
    #                                      high_bias=False,
    #                                      exp_scale=0.1)
    # high_bias_data = t.meta_training(task_params,
    #                                  arm_set,
    #                                  estim=init_estim,
    #                                  method='ccipca',
    #                                  high_bias=True,
    #                                  exp_scale=0.1)
    peleg_data = t.meta_training(task_params,
                                arm_set,
                                estim=init_estim,
                                method='meta_prior',
                                decision_making='meta_prior',
                                dim_known=False,
                                exp_scale=1)
    ts_data = t.meta_training(task_params,
                              arm_set,
                              estim=init_estim,
                              method='ccipca',
                              decision_making='ts',
                              dim_known=False,
                              exp_scale=1)
    ipca_dimunknown_data = t.meta_training(task_params,
                                           arm_set,
                                           estim=init_estim,
                                           method='ccipca',
                                           exp_scale=1,
                                           dim_known=False)
    cella_data = t.meta_training(task_params,
                                 arm_set,
                                 estim=init_estim,
                                 method='full_dimension',
                                 exp_scale=1,
                                 dim_known=False)
    linucb_data = t.meta_training(task_params,
                                       arm_set,
                                       estim=init_estim,
                                       method='classic_learning',
                                       exp_scale=1)
    thomps_data = t.meta_training(task_params,
                                       arm_set,
                                       estim=init_estim,
                                       method='classic_learning',
                                       exp_scale=1,
                                       decision_making='ts')
    ideal_data = t.meta_training(task_params,
                                arm_set,
                                estim=init_estim,
                                method='ideal_learning',
                                exp_scale=1,
                                ideal_proj=real_proj,
                                ideal_offset=off_set,
                                decision_making='ucb')

    e.multiple_regret_plots([linucb_data[0]],
                            [linucb_data[1]],
                            plot_label="LinUCB")
    e.multiple_regret_plots([thomps_data[0]],
                            [thomps_data[1]],
                            plot_label="Thompson Sampling")
    e.multiple_regret_plots([cella_data[0]],
                            [cella_data[1]],
                            plot_label="Cella et al., 2020")
    e.multiple_regret_plots([ideal_data[0]],
                            [ideal_data[1]],
                            plot_label="Oracle Projection")
    # e.multiple_regret_plots([high_varianca_data[0]],
    #                         [high_varianca_data[1]],
    #                         plot_label="High Variance Solution")
    # e.multiple_regret_plots([high_bias_data[0]],
    #                         [high_bias_data[1]],
    #                         plot_label="High Bias Solution",
    #                         do_plot=True)
    e.multiple_regret_plots([ts_data[0]],
                            [ts_data[1]],
                            plot_label="Projected Thompson Sampling")
    e.multiple_regret_plots([ipca_dimunknown_data[0]],
                            [ipca_dimunknown_data[1]],
                            plot_label="Projected LinUCB")
    e.multiple_regret_plots([peleg_data[0]],
                            [peleg_data[1]],
                            plot_label="Peleg et al., 2022",
                            do_plot=True)


def multi_task_evaluation(task_params, arm_set, init_estim, proj_mat):
    '''Evaluates multi task regret, given the projection.'''
    proj_regret_per_task = np.zeros(c.EPOCHS)
    linucb_regret_per_task = np.zeros(c.EPOCHS)
    proj_std_per_task = np.zeros(c.EPOCHS)
    linucb_std_per_task = np.zeros(c.EPOCHS)

    for i in np.arange(len(task_params)):
        linucb_data = t.projected_training(task_params[i],
                                           arm_set,
                                           estim=init_estim,
                                           exp_scale=0.1)
        linucb_regret_per_task += linucb_data[0]/c.NO_TASK
        linucb_std_per_task += linucb_data[1]**2/c.NO_TASK
    linucb_std_per_task = np.sqrt(linucb_std_per_task)

    for i in np.arange(len(task_params)):
        proj_data = t.projected_training(task_params[i],
                                         arm_set,
                                         proj_mat=proj_mat,
                                         estim=init_estim,
                                         exp_scale=0.1)
        proj_regret_per_task += proj_data[0]/c.NO_TASK
        proj_std_per_task += proj_data[1]**2/c.NO_TASK
    proj_std_per_task = np.sqrt(proj_std_per_task)

    e.multiple_regret_plots([linucb_regret_per_task],
                            [linucb_std_per_task],
                            plot_label="LinUCB")
    e.multiple_regret_plots([proj_regret_per_task],
                            [proj_std_per_task],
                            plot_label="Projected Bandits",
                            do_plot=True)

if __name__ == '__main__':
    main()
