import numpy as np
import synthetic_data as sd
import config as c
import training as t
import evaluation as e


def main():
    '''Main function used to evaluate different bandit settings.'''

    multi_tasks = sd.TaskData()
    arm_set = multi_tasks.target_context
    task_params = multi_tasks.theta_opt
    proj_mat = multi_tasks.subspace_projection
    off_set = np.tile(multi_tasks.off_set, (c.REPEATS, 1))
    proj_mat = np.tile(proj_mat, (c.REPEATS, 1, 1))
    init_estim = np.abs(np.random.uniform(size=c.DIMENSION))

    meta_learning_evaluation(task_params, arm_set, init_estim, proj_mat, off_set)
    # multi_task_evaluation(task_params, arm_set, init_estim, proj_mat)


def meta_learning_evaluation(task_params, arm_set, init_estim, real_proj, off_set):
    '''Evaluation of the meta learning task.'''

    ipca_regret, ipca_std = t.meta_training(task_params,
                                            arm_set,
                                            estim=init_estim,
                                            method='ccipca',
                                            exp_scale=0.1)
    ts_regret, ts_std = t.meta_training(task_params,
                                        arm_set,
                                        estim=init_estim,
                                        method='ccipca',
                                        decision_making='ts',
                                        dim_known=True,
                                        exp_scale=0.1)
    ipca_dimknown_regret, ipca_dimknown_std = t.meta_training(task_params,
                                                              arm_set,
                                                              estim=init_estim,
                                                              method='ccipca',
                                                              exp_scale=0.1,
                                                              dim_known=True)
    # proj_regret, proj_std = t.meta_training(task_params,
    #                                         arm_set,
    #                                         estim=init_estim,
    #                                         method='sga',
    #                                         exp_scale=0.1)
    linucb_regret, linucb_std, theta_estim = t.projected_training(task_params[-1],
                                                                  arm_set,
                                                                  estim=init_estim,
                                                                  exp_scale=0.1)
    thomps_regret, thomps_std, theta_estim = t.projected_training(task_params[-1],
                                                                  arm_set,
                                                                  estim=init_estim,
                                                                  exp_scale=0.1,
                                                                  decision_making='ts')
    ideal_regret, ideal_std, theta = t.projected_training(task_params[-1],
                                                          arm_set,
                                                          proj_mat=real_proj,
                                                          off_set=off_set,
                                                          estim=init_estim,
                                                          exp_scale=0.1,
                                                          decision_making='ucb')

    e.multiple_regret_plots([linucb_regret],
                            [linucb_std],
                            plot_label="LinUCB")
    e.multiple_regret_plots([thomps_regret],
                            [thomps_std],
                            plot_label="Thompson Sampling")
    e.multiple_regret_plots([ideal_regret],
                            [ideal_std],
                            plot_label="Real Projection LinUCB")
    e.multiple_regret_plots([ipca_regret],
                            [ipca_std],
                            plot_label="Projected LinUCB")
    e.multiple_regret_plots([ts_regret],
                            [ts_std],
                            plot_label="Projected Thompson Sampling")
    e.multiple_regret_plots([ipca_dimknown_regret],
                            [ipca_dimknown_std],
                            plot_label="Projected LinUCB Rank known",
                            do_plot=True)
    # e.multiple_regret_plots([proj_regret],
    #                         [proj_std],
    #                         plot_label="SGA",
    #                         do_plot=True)


def multi_task_evaluation(task_params, arm_set, init_estim, proj_mat):
    '''Evaluates multi task regret, given the projection.'''
    proj_regret_per_task = np.zeros(c.EPOCHS)
    linucb_regret_per_task = np.zeros(c.EPOCHS)
    proj_std_per_task = np.zeros(c.EPOCHS)
    linucb_std_per_task = np.zeros(c.EPOCHS)

    for i in np.arange(len(task_params)):
        linucb_regret, linucb_std, theta = t.projected_training(task_params[i],
                                                                arm_set,
                                                                estim=init_estim,
                                                                exp_scale=0.1)
        linucb_regret_per_task += linucb_regret/c.NO_TASK
        linucb_std_per_task += linucb_std**2/c.NO_TASK
    linucb_std_per_task = np.sqrt(linucb_std_per_task)

    for i in np.arange(len(task_params)):
        proj_regret, proj_std, theta = t.projected_training(task_params[i],
                                                            arm_set,
                                                            proj_mat=proj_mat,
                                                            estim=init_estim,
                                                            exp_scale=0.1)
        proj_regret_per_task += proj_regret/c.NO_TASK
        proj_std_per_task += proj_std**2/c.NO_TASK
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
