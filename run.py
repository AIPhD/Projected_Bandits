import random
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
    # print(np.max(np.sqrt(np.einsum('ij,ij->i', task_params, task_params))))
    # print(np.min(np.sqrt(np.einsum('ij,ij->i', task_params, task_params))))

    filter_data = [
                   ['F', 18, '0', True],
                   ['F', 18, '0', False],
                   ['F', 35, '12', True],
                   ['F', 35, '12', False],
                   ['F', 25, '4', False],
                   ['F', 25, '4', True],
                   ['F', 35, '6', False],
                   ['F', 35, '6', True],
                   ['F', 35, '9', False],
                   ['F', 35, '9', True],
                #    ['F', 35, '11', True],
                   ['F', 35, '11', False],
                   ['F', 35, '19', False],
                   ['F', 35, '19', True],
                   ['F', 45, '15', True],
                   ['F', 45, '15', False],
                   ['M', 18, '0', True],
                   ['M', 18, '0', False],
                   ['M', 35, '2', False],
                   ['M', 35, '2', True],
                   ['M', 35, '3', False],
                   ['M', 35, '3', True],
                #    ['M', 25, '12', False],
                #    ['M', 35, '11', False],
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

    # for i in range(9,18):
    #     real_data_comparison(gender=None, age=None, prof=str(i))
    real_data_comparison(gender='M', age=None, prof=None, max_tasks=400, y_top_limit= 40)
    # real_regret_per_dimension_eval(gender='M', age=None, prof=None, y_top_limit=40)    
    # regret_per_dimension_eval(task_params, arm_set, init_estim)
    # meta_learning_evaluation(task_params, arm_set, init_estim, proj_mat, off_set)
    # lin_ucb_comparison(task_params, arm_set, init_estim)
    # full_dim_comparison(task_params, arm_set, init_estim)
    # w_error_eval(task_params, arm_set, init_estim, proj_mat, off_set)
    # multi_task_evaluation(task_params, arm_set, init_estim, proj_mat)


def real_data_comparison(gender=None,
                         age=None,
                         prof=None,
                         max_tasks=150,
                         context=rd.context_data_set,
                         rewards=rd.reward_data_set,
                         y_top_limit=None):
    '''Compares multiple algorithms on real data'''

    dimension = len(context[0])
    explore_scale = 0.1
    filtered_users = rd.filter_users(np.asarray(rd.user_data_set),
                                     gender=gender,
                                     age=age,
                                     prof=prof)
    filtered_user_index = np.random.choice(np.asarray([int(i) for i in filtered_users[:, 0]]) - 1, size=max_tasks, replace=False)
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
    # thomps_data = rt.real_meta_training(filtered_user_index,
    #                                     context,
    #                                     rewards,
    #                                     dimension,
    #                                     method="classic_learning",
    #                                     exp_scale=explore_scale,
    #                                     decision_making='ts')

    e.multiple_regret_plots([linucb_data[0]],
                            [linucb_data[1]],
                            plot_label="LinUCB",
                            y_top_limit=y_top_limit)
    # e.multiple_regret_plots([thomps_data[0]],
    #                         [thomps_data[1]],
    #                         plot_label="Thompson Sampling",
    #                         y_top_limit=y_top_limit)
    e.multiple_regret_plots([cella_data[0]],
                            [cella_data[1]],
                            plot_label="B-OFUL")
    e.multiple_regret_plots([ts_data[0]],
                            [ts_data[1]],
                            plot_label="P-TS",
                            y_top_limit=y_top_limit)
    e.multiple_regret_plots([ipca_dimunknown_data[0]],
                            [ipca_dimunknown_data[1]],
                            plot_label="P-LinUCB",
                            directory="real_data_experiments",
                            plotsuffix=f'real_regret_comparison_{gender}_{age}_{prof}',
                            y_label="Expected Transfer Regret",
                            do_plot=True,
                            y_top_limit=y_top_limit)

    e.multiple_regret_plots([linucb_data[2]],
                            [linucb_data[3]],
                            plot_label="LinUCB")
    # e.multiple_regret_plots([thomps_data[2]],
    #                         [thomps_data[3]],
    #                         plot_label="Thompson Sampling")
    e.multiple_regret_plots([cella_data[2]],
                            [cella_data[3]],
                            plot_label="B-OFUL")
    e.multiple_regret_plots([ts_data[2]],
                            [ts_data[3]],
                            plot_label="P-TS")
    e.multiple_regret_plots([ipca_dimunknown_data[2]],
                            [ipca_dimunknown_data[3]],
                            plot_label="P-LinUCB",
                            directory="real_data_experiments",
                            plotsuffix=f'real_meta_regret_comparison_{gender}',
                            y_label='Cumulative Regret',
                            x_label='Number of Users',
                            do_plot=True)


def regret_per_dimension_eval(task_params, arm_set, init_estim):
    '''Evaluate total accumulated regret as a function of q for synthetic data.'''

    ranks = np.arange(31) # np.asarray([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
    total_regret_ts_dimensional = np.zeros(len(ranks))
    total_regret_linucb_dimensional = np.zeros(len(ranks))
    total_error_linucb_dimensional = np.zeros(len(ranks))
    total_error_ts_dimensional = np.zeros(len(ranks))

    i=0
    for d in ranks:
        ts_data = t.meta_training(task_params,
                                  arm_set,
                                  estim=init_estim,
                                  method='ccipca',
                                  decision_making='ts',
                                  dim_known=True,
                                  exp_scale=1,
                                  dim_set=d)
        total_regret_ts_dimensional[i]= ts_data[2][-1]
        total_error_ts_dimensional[i]= ts_data[3][-1]

        linucb_data = t.meta_training(task_params,
                                      arm_set,
                                      estim=init_estim,
                                      method='ccipca',
                                      exp_scale=1,
                                      dim_known=True,
                                      dim_set=d)
        total_regret_linucb_dimensional[i]= linucb_data[2][-1]
        total_error_linucb_dimensional[i]= linucb_data[3][-1]
        print(str(d) +' dimensions align.')
        i += 1

    e.dimensional_regret_plots(total_regret_ts_dimensional,
                               x_scale=ranks,
                               errors=total_error_ts_dimensional,
                               plot_label='P-TS')
    e.dimensional_regret_plots(total_regret_linucb_dimensional,
                               x_scale=ranks,
                               errors=total_error_linucb_dimensional,
                               plot_label='P-LinUCB',
                               plotsuffix='meta_dimensional_regret',
                               directory='synthetic_data_experiments',
                               y_label="Total Accumulated Regret",
                               do_plot=True)



def real_regret_per_dimension_eval(gender=None,
                                   age=None,
                                   prof=None,
                                   max_tasks=150,
                                   context=rd.context_data_set,
                                   rewards=rd.reward_data_set,
                                   y_top_limit=None):
    '''Evaluate total accumulated regret as a function of q for real data.'''

    dimension = len(context[0])
    total_regret_ts_dimensional = np.zeros(dimension+1)
    total_regret_linucb_dimensional = np.zeros(dimension+1)
    total_error_linucb_dimensional = np.zeros(dimension+1)
    total_error_ts_dimensional = np.zeros(dimension+1)
    explore_scale = 0.1
    filtered_users = rd.filter_users(np.asarray(rd.user_data_set),
                                     gender=gender,
                                     age=age,
                                     prof=prof)
    filtered_user_index = np.random.choice(np.asarray([int(i) for i in filtered_users[:, 0]]) - 1, size=max_tasks, replace=False)

    for d in np.arange(dimension+1):
        ts_data = rt.real_meta_training(filtered_user_index,
                                   context,
                                   rewards,
                                   dimension,
                                   method='ccipca',
                                   decision_making='ts',
                                   dim_known=True,
                                   exp_scale=explore_scale,
                                   dim_set=d)
        total_regret_ts_dimensional[d]= ts_data[0][-1]
        total_error_ts_dimensional[d]= ts_data[1][-1]

        linucb_data = rt.real_meta_training(filtered_user_index,
                                       context,
                                       rewards,
                                       dimension,
                                       method='ccipca',
                                       exp_scale=explore_scale,
                                       dim_known=True,
                                       dim_set=d)
        total_regret_linucb_dimensional[d]= linucb_data[0][-1]
        total_error_linucb_dimensional[d]= linucb_data[1][-1]
        print(str(d) +' dimensions align.')
        
    e.dimensional_regret_plots(total_regret_ts_dimensional,
                               errors=total_error_ts_dimensional,
                               y_top_limit=y_top_limit,
                               plot_label='Projected Thompson Sampling')
    e.dimensional_regret_plots(total_regret_linucb_dimensional,
                               errors=total_error_linucb_dimensional,
                               plotsuffix='Real_dim_regret_comparison',
                               plot_label='Projected LinUCB',
                               y_top_limit=y_top_limit,
                               do_plot=True)


def w_error_eval(task_params, arm_set, init_estim, real_proj, off_set):
    '''Plot W error for ts and linucb'''

    w_mean = 1/c.NO_TASK * np.sum(task_params, axis=0) - task_params
    w_2 = np.einsum('ij,ij->i', w_mean, w_mean)
    expected_w2 = 1/c.NO_TASK * np.sum(w_2, axis=0)
    ts_data = t.meta_training(task_params,
                              arm_set,
                              estim=init_estim,
                              method='ccipca',
                              decision_making='ts',
                              dim_known=True,
                              dim_set=c.DIMENSION_ALIGN,
                              exp_scale=1)
    ipca_dimunknown_data = t.meta_training(task_params,
                                           arm_set,
                                           estim=init_estim,
                                           method='ccipca',
                                           exp_scale=1,
                                           dim_known=True,
                                           dim_set=c.DIMENSION_ALIGN)
    no_meta_data = t.meta_training(task_params,
                                   arm_set,
                                   estim=init_estim,
                                   method="ccipca",
                                   dim_known=True,
                                   dim_set=c.DIMENSION)
    ideal_data = c.KAPPA * np.ones(c.NO_TASK-c.TASK_INIT)

    e.projection_error_plot(np.abs(ipca_dimunknown_data[4]/c.KAPPA-1),
                            plot_label="P-TS")
    e.projection_error_plot(np.abs(ts_data[4]/c.KAPPA-1),
                            plot_label="P-LinUCB")
    e.projection_error_plot(np.abs(no_meta_data[4]/c.KAPPA-1),
                            plot_label = "B-OFUL",
                            do_plot=True)
    # e.projection_error_plot(expected_w2 * np.ones(c.NO_TASK-c.TASK_INIT),
    #                         plot_label = r'$\mathrm{Var}_{\max}$',
    #                         do_plot=True)
    # e.projection_error_plot(ideal_data/c.KAPPA,
    #                         plot_label=r"$\mathrm{Var}_{\rho}$",
    #                         do_plot=True)

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
    ts_data = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='ccipca',
                                    decision_making='ts',
                                    dim_known=False,
                                    exp_scale=1)
    peleg_data = t.meta_training(task_params,
                                arm_set,
                                estim=init_estim,
                                method='meta_prior',
                                decision_making='meta_prior',
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
                                method='ideal_proj',
                                exp_scale=1,
                                ideal_proj=real_proj,
                                ideal_offset=off_set,
                                decision_making='ucb')

    e.multiple_regret_plots([linucb_data[0]],
                            [linucb_data[1]],
                            plot_label="LinUCB")
    e.multiple_regret_plots([thomps_data[0]],
                            [thomps_data[1]],
                            plot_label="TS")
    e.multiple_regret_plots([cella_data[0]],
                            [cella_data[1]],
                            plot_label="B-OFUL")
    # e.multiple_regret_plots([high_varianca_data[0]],
    #                         [high_varianca_data[1]],
    #                         plot_label="High Variance Solution")
    # e.multiple_regret_plots([high_bias_data[0]],
    #                         [high_bias_data[1]],
    #                         plot_label="High Bias Solution",
    #                         do_plot=True)
    e.multiple_regret_plots([ts_data[0]],
                            [ts_data[1]],
                            plot_label="P-TS")
    e.multiple_regret_plots([ipca_dimunknown_data[0]],
                            [ipca_dimunknown_data[1]],
                            plot_label="P-LinUCB")
    e.multiple_regret_plots([ideal_data[0]],
                            [ideal_data[1]],
                            plot_label="Oracle")
    e.multiple_regret_plots([peleg_data[0]],
                            [peleg_data[1]],
                            plot_label="M-TS",
                            do_plot=True,
                            directory="synthetic_data_experiments",
                            plotsuffix='synthetic_expected_regret_comparison',
                            y_label="Expected Transfer Regret")

    e.multiple_regret_plots([linucb_data[2]],
                            [linucb_data[3]],
                            plot_label="LinUCB")
    e.multiple_regret_plots([thomps_data[2]],
                            [thomps_data[3]],
                            plot_label="TS")
    e.multiple_regret_plots([cella_data[2]],
                            [cella_data[3]],
                            plot_label="B-OFUL")
    e.multiple_regret_plots([ts_data[2]],
                            [ts_data[3]],
                            plot_label="P-TS")
    e.multiple_regret_plots([ipca_dimunknown_data[2]],
                            [ipca_dimunknown_data[3]],
                            plot_label="P-LinUCB")
    e.multiple_regret_plots([ideal_data[2]],
                            [ideal_data[3]],
                            plot_label="Oracle")
    e.multiple_regret_plots([peleg_data[2]],
                            [peleg_data[3]],
                            plot_label="M-TS",
                            directory="synthetic_data_experiments",
                            plotsuffix='synthetic_meta_regret_comparison',
                            y_label='Cumulative Regret',
                            x_label='Number of Tasks',
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

def lin_ucb_comparison(task_params, arm_set, init_estim):
    '''Hyper parameter comparison for classic LinUCB'''
    linucb_data_1 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='classic_learning',
                                    exp_scale=1,
                                    lamb_1=0,
                                    lamb_2=10)
    linucb_data_2 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='classic_learning',
                                    exp_scale=1,
                                    lamb_1=0,
                                    lamb_2=6)
    linucb_data_3 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='classic_learning',
                                    exp_scale=1,
                                    lamb_1=0,
                                    lamb_2=30)
    linucb_data_4 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='classic_learning',
                                    exp_scale=1,
                                    lamb_1=0,
                                    lamb_2=40)
    linucb_data_5 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='classic_learning',
                                    exp_scale=1,
                                    lamb_1=0,
                                    lamb_2=50)
    

    e.multiple_regret_plots([linucb_data_1[0]],
                            [linucb_data_1[1]],
                            plot_label=r"LinUCB $\lambda=10$")
    e.multiple_regret_plots([linucb_data_2[0]],
                            [linucb_data_2[1]],
                            plot_label=r"LinUCB $\lambda=6$")
    e.multiple_regret_plots([linucb_data_3[0]],
                            [linucb_data_3[1]],
                            plot_label=r"LinUCB $\lambda=30$")
    e.multiple_regret_plots([linucb_data_4[0]],
                            [linucb_data_4[1]],
                            plot_label=r"LinUCB $\lambda=40$")
    e.multiple_regret_plots([linucb_data_5[0]],
                            [linucb_data_5[1]],
                            plot_label=r"LinUCB $\lambda=50$",
                            do_plot=True,
                            directory="synthetic_data_experiments",
                            plotsuffix='synthetic_expected_regret_linucb_param_comparison',
                            y_label="Expected Transfer Regret")


def full_dim_comparison(task_params, arm_set, init_estim):
    '''Hyper parameter comparison for classic LinUCB'''

    full_dim_data_1 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='full_dimension',
                                    exp_scale=1,
                                    lamb_2=1,
                                    lamb_1=10)
    full_dim_data_2 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='full_dimension',
                                    exp_scale=1,
                                    lamb_2=1,
                                    lamb_1=20)
    full_dim_data_3 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='full_dimension',
                                    exp_scale=1,
                                    lamb_2=1,
                                    lamb_1=30)
    full_dim_data_4 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='full_dimension',
                                    exp_scale=1,
                                    lamb_2=1,
                                    lamb_1=40)
    full_dim_data_5 = t.meta_training(task_params,
                                    arm_set,
                                    estim=init_estim,
                                    method='full_dimension',
                                    exp_scale=1,
                                    lamb_2=1,
                                    lamb_1=100)
    

    e.multiple_regret_plots([full_dim_data_1[0]],
                            [full_dim_data_1[1]],
                            plot_label=r"B-OFUL $\lambda=10$")
    e.multiple_regret_plots([full_dim_data_2[0]],
                            [full_dim_data_2[1]],
                            plot_label=r"B-OFUL $\lambda=20$")
    e.multiple_regret_plots([full_dim_data_3[0]],
                            [full_dim_data_3[1]],
                            plot_label=r"B-OFUL $\lambda=30$")
    e.multiple_regret_plots([full_dim_data_4[0]],
                            [full_dim_data_4[1]],
                            plot_label=r"B-OFUL $\lambda=40$")
    e.multiple_regret_plots([full_dim_data_5[0]],
                            [full_dim_data_5[1]],
                            plot_label=r"B-OFUL $\lambda=100$",
                            do_plot=True,
                            directory="synthetic_data_experiments",
                            plotsuffix='synthetic_expected_regret_fulldim_param_comparison',
                            y_label="Expected Transfer Regret")


if __name__ == '__main__':
    main()
