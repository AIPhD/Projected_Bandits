import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config as c
matplotlib.use('TKAgg')

font  = {
    'size' : 14
}

PLOT_DIR = '/home/steven/Projected_Bandits/plots/'

plt.rcParams.update({
    "text.usetex": True,
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
                             errors=None,
                             plot_label=None,
                             plotsuffix='dimensional_regret_comparison',
                             directory='test',
                             do_plot=False,
                             y_label='Total Expected Transfer Regret',
                             y_top_limit=None):
    '''Plot multiple regret evolutions for comparison.'''

    i = 0
    plt.xlabel('rank q')
    plt.ylabel(y_label)
    if y_top_limit is not None:
        plt.ylim(0, y_top_limit)
 
    x_scale = np.cumsum(np.ones(len(regret)))-1
   
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
