import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config as c
matplotlib.use('TKAgg')

font  = {
    'size' : 22
}

PLOT_DIR = '/home/steven/Projected_Bandits/plots/'

plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{amsfonts, amsmath}',
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "savefig.dpi": 800})

matplotlib.rc('font', **font)


def multiple_regret_plots(regrets,
                          errors=None,
                          bethas=None,
                          plot_label=None,
                          plotsuffix='regret_comparison',
                          directory='test',
                          do_plot=False,
                          y_label='Regret',
                          x_label='Rounds',
                          y_top_limit=None):
    '''Plot multiple regret evolutions for comparison.'''

    i = 0
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    x_scale = np.cumsum(np.ones(len(regrets[0])))
    if y_top_limit is not None:
        plt.ylim(0, y_top_limit)

    if bethas is None:
        for regret in regrets:
            if errors is not None:
                plt.errorbar(x_scale,
                             regret,
                             yerr=errors[i],
                             label=plot_label,
                             alpha=0.75,
                             errorevery=49)
                # plt.plot(np.arange(c.EPOCHS),
                #          regret + errors[i],
                #          alpha=0.25,
                #          ls='--')
                # plt.plot(np.arange(c.EPOCHS),
                #          regret - errors[i],
                #          alpha=0.25,
                #          ls='--')
                i += 1

            else:
                plt.plot(np.arange(c.EPOCHS),
                         regret,
                         label=plot_label)

    else:
        for regret in regrets:
            beta=bethas[i]
            if errors is not None:
                plt.errorbar(np.cumsum(np.ones(c.EPOCHS)),
                                 regret,
                                 yerr=errors[i],
                                 label=f"{plot_label}"+r" $\beta$ = "+f"{beta}",
                                 alpha=0.75,
                                 errorevery=49)
                # plt.plot(np.arange(c.EPOCHS),
                #          regret + errors[i],
                #          alpha=0.25,
                #          ls='--')
                # plt.plot(np.arange(c.EPOCHS),
                #          regret - errors[i],
                #          alpha=0.25,
                #          ls='--')

            else:
                plt.plot(np.arange(c.EPOCHS),
                         regret,
                         label=f"{plot_label}"+r" $\beta$ = "+f"{beta}")

            i += 1

    plt.legend()
    plt.tight_layout()

    if do_plot:
        plt.savefig(f'{PLOT_DIR}{directory}/{plotsuffix}.pdf',
                    dpi=800,
                    format= "pdf",
                    bbox_inches='tight',
                    pad_inches=0)
        plt.show()
        plt.close()


def dimensional_regret_plots(regret,
                             x_scale = np.cumsum(np.ones(c.DIMENSION))-1,
                             errors=None,
                             plot_label=None,
                             plotsuffix='dimensional_regret_comparison',
                             directory='test',
                             do_plot=False,
                             y_label='Total Expected Transfer Regret',
                             y_top_limit=None):
    '''Plot multiple regret evolutions for comparison.'''

    plt.xlabel(r'$q$')
    plt.ylabel(y_label)
    if y_top_limit is not None:
        plt.ylim(0, y_top_limit)



    plt.errorbar(x_scale,
                 regret,
                 yerr=errors,
                 label=plot_label,
                 alpha=0.75,
                 errorevery=2)
    # plt.plot(np.arange(c.EPOCHS),
    #          regret + errors[i],
    #          alpha=0.25,
    #          ls='--')
    # plt.plot(np.arange(c.EPOCHS),
    #          regret - errors[i],
    #          alpha=0.25,
    #          ls='--')


    plt.legend()
    plt.tight_layout()

    if do_plot:
        plt.savefig(f'{PLOT_DIR}{directory}/{plotsuffix}.pdf',
                    dpi=800,
                    format= "pdf",
                    bbox_inches='tight',
                    pad_inches=0)
        plt.show()
        plt.close()


def projection_error_plot(w_array,
                          errors=None,
                          plot_label=None,
                          plotsuffix='w_error_plot',
                          directory='synthetic_data_experiments',
                          do_plot=False,
                          y_label=r'$|\mathbb{E}_{\boldsymbol{\theta}^{*}\sim\rho}[\mathbb{E}[W]]^2/\mathrm{Var}_{\rho}-1|$',
                          y_top_limit=None):
    '''Plot the expected projection loss over tasks.'''

    plt.xlabel('Number of Tasks')
    plt.ylabel(y_label)
    plt.yscale('log', nonposy='clip')
    if y_top_limit is not None:
        plt.ylim(0, y_top_limit)

    x_scale = np.cumsum(np.ones(len(w_array)))-1

    plt.errorbar(x_scale + c.TASK_INIT,
                 w_array,
                 yerr=errors,
                 label=plot_label,
                 alpha=0.75,
                 errorevery=2)
    # plt.plot(np.arange(c.EPOCHS),
    #          regret + errors[i],
    #          alpha=0.25,
    #          ls='--')
    # plt.plot(np.arange(c.EPOCHS),
    #          regret - errors[i],
    #          alpha=0.25,
    #          ls='--')


    plt.legend()
    plt.tight_layout()

    if do_plot:
        plt.savefig(f'{PLOT_DIR}{directory}/{plotsuffix}.pdf',
                    dpi=800,
                    format= "pdf",
                    bbox_inches='tight',
                    pad_inches=0)
        plt.show()
        plt.close()
