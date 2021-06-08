#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
"""Figure functions for the imported cases paper.

These are fixed, static figures, not part of the web app.
"""

import datetime
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np


def plot_forward_simulations(import_cases,
                             R_t_times,
                             R_t,
                             epsilons,
                             simulated_case_data,
                             first_day,
                             show=True):
    """Make a figure showing simulated cases for various values of epsilon.

    It has three panels:
        a. Number of imported cases on each day.
        b. R_t over time.
        c. Simulated local cases for different values of epsilon.

    Parameters
    ----------
    import_cases : list of int
        Daily incident imported cases, starting on first_day + 1day
    R_t_times : list of int
        List of integer time points on which R_t is defined, relative to
        first_day
    R_t : list of float
        Trajectory of reproduction number (local)
    epsilons : list of float
        Values of epsilon for which local cases were simulated
    simulated_case_data : list of pandas.DataFrame
        For each epsilon, a dataframe giving the simulated local cases. Each
        dataframe should have the following three columns: 'Mean',
        'Lower bound CI', and 'Upper bound CI'.
    first_day : datetime.datetime
        The first day for simulated local data and imported data
    show : bool, optional (True)
        Whether or not to plt.show() the figure after it has been generated

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Line styles for each value of epsilon. Currently supports up to three
    # epsilons.
    epsilon_colors = ['forestgreen', 'tab:pink', 'tan']
    styles = ['-', '-.', '--']

    # Define all three axes to have the same x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, gridspec_kw={'height_ratios': [1, 1, 2.5]}, sharex=True)

    # Date times for the simulated local cases
    date_times = [first_day + datetime.timedelta(days=int(i))
                  for i in np.arange(len(simulated_case_data[0]))]

    # Date times for imported cases. They can end before the simulations
    import_times = date_times[1:len(import_cases)+1]

    # Date times for R_t. R_t can be defined on a subset of the time
    R_t_times = date_times[R_t_times[0]:R_t_times[-1]+1]

    # Plot bars for imported cases
    ax1.bar(import_times,
            import_cases,
            color='red',
            hatch='/////',
            edgecolor='w',
            lw=0.1)

    ax1.set_ylabel('Imported\n cases')
    ax1.tick_params(labelbottom=False)

    # Plot line for R_t
    ax2.plot(R_t_times, R_t, color='k')
    ax2.axhline(1, ls='-', color='gray', lw=1.5, alpha=0.4, zorder=-10)
    ax2.set_ylabel(r'$R_t^\mathrm{local}$')
    ax2.set_ylim(0, 1.1*max(R_t))
    ax2.tick_params(labelbottom=False)

    # Plot shaded regions for simulated cases
    legend_entries = []
    legend_labels = []
    for ls, color, epsilon, df in zip(styles,
                                      epsilon_colors,
                                      epsilons,
                                      simulated_case_data):
        line, = ax3.plot(date_times, df['Mean'], color=color, ls=ls, lw=2)
        shade = ax3.fill_between(date_times,
                                 df['Lower bound CI'],
                                 df['Upper bound CI'],
                                 alpha=0.25,
                                 color=color)
        legend_entries.append((line, shade))
        legend_labels.append(r'$ϵ={}$'.format(epsilon))

    ax3.set_ylabel('Simulated local cases')
    ax3.legend(legend_entries, legend_labels, loc='upper left')

    # Set ticks once per week
    ax3.set_xticks(date_times[::7])

    # Use "Jan 01", etc as the date format
    ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
    plt.xticks(rotation=45, ha='center')

    # Resize the figure and add panel labels (a), (b), (c)
    fig.set_size_inches(5, 6.5)
    fig.text(0.025, 0.9, '(a)', fontsize=14)
    fig.text(0.025, 0.68, '(b)', fontsize=14)
    fig.text(0.025, 0.475, '(c)', fontsize=14)

    # Manual tuning of plot size
    plt.subplots_adjust(top=0.88,
                        bottom=0.11,
                        left=0.155,
                        right=0.9,
                        hspace=0.31,
                        wspace=0.2)

    if show:
        plt.show()

    return fig


def plot_r_inference(first_day_data,
                     local_cases,
                     import_cases,
                     first_day_inference,
                     epsilons,
                     R_t_results,
                     prior_mid,
                     default_epsilon=0,
                     show=True):
    """Make a figure showing R_t inference for different choices of epsilon.

    It has two panels:
        a. Local and imported cases which were used for inference
        b. Subplots each comparing R_t for one choice of epsilon with the
           default choice.

    Notes
    -----
    As written this function expects a total of five epsilon values (including
    the default value).

    Parameters
    ----------
    first_day_data : datetime.datetime
        First day of incidence data
    local_cases : list of int
        Daily incident local cases
    import_cases : list of int
        Daily incident imported cases
    first_day_inference : datetime.datetime
        First day of inference results
    epsilons : list of float
        Values of epsilon for which inference was performed
    R_t_results : list of pandas.DataFrame
        For each epsilon, a dataframe giving the inference results for R_t. It
        must have the three columns 'Mean', 'Lower bound CI', and
        'Upper bound CI'.
    prior_mid : float
        The prior median of R_t
    default_epsilon : float, optional (0)
        The value of epsilon whose inference results will be compared to the
        results from all other values of epsilon.
    show : bool, optional (True)
        Whether or not to plt.show() the figure after it has been generated

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Build grid of subplots
    # Use 0.1 height ratio subplot rows to space out the panels
    fig = plt.figure()
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 0.1, 1, 1])

    # Ax for case data
    top_ax = fig.add_subplot(gs[0, :])

    # Axes for R_t inference
    axs = [fig.add_subplot(gs[i, j]) for i in [2, 3] for j in [0, 1]]

    # Make them all share both x and y axis
    axs[1].sharex(axs[0])
    axs[2].sharex(axs[0])
    axs[3].sharex(axs[0])
    axs[1].sharey(axs[0])
    axs[2].sharey(axs[0])
    axs[3].sharey(axs[0])
    axs[0].tick_params(labelbottom=False)
    axs[1].tick_params(labelbottom=False)
    axs[1].tick_params(labelleft=False)
    axs[3].tick_params(labelleft=False)

    # Plot local and imported cases
    width = datetime.timedelta(hours=10)
    data_times = [first_day_data + datetime.timedelta(days=int(i))
                  for i in range(len(local_cases))]
    top_ax.bar([x - width/2 for x in data_times],
               local_cases,
               width,
               label='Local cases',
               color='k',
               alpha=0.8)
    top_ax.bar([x + width/2 for x in data_times],
               import_cases,
               width,
               hatch='/////',
               edgecolor='w',
               lw=0.1,
               label='Imported cases',
               color='red')
    top_ax.legend()

    # Get R_t for the default epsilon
    default_results = R_t_results[epsilons.index(default_epsilon)]

    # Build time vector for all R_t
    times = len(default_results['Mean'])
    date_times = [first_day_inference + datetime.timedelta(days=int(i))
                  for i in range(times)]

    i = 0
    for epsilon, results in zip(epsilons, R_t_results):
        if epsilon != default_epsilon:
            ax = axs[i]

            # Plot shaded region for R_t
            line, = ax.plot(date_times,
                            results['Mean'],
                            color='royalblue',
                            lw=1.0,
                            zorder=7)
            shade = ax.fill_between(date_times,
                                    results['Lower bound CI'],
                                    results['Upper bound CI'],
                                    alpha=0.3,
                                    color='royalblue',
                                    zorder=6,
                                    linewidth=0.0)

            # Plot another region for the default epsilon inference results
            zeroline, = ax.plot(date_times,
                                default_results['Mean'],
                                color='k',
                                lw=1.0,
                                ls='--',
                                zorder=10)
            zerorange = ax.fill_between(date_times,
                                        default_results['Lower bound CI'],
                                        default_results['Upper bound CI'],
                                        alpha=0.35,
                                        color='k',
                                        zorder=-10,
                                        linewidth=0.0)
            # Add a texture to the region for default epsilon R_t
            zerorangelines = ax.fill_between(
                date_times,
                default_results['Lower bound CI'],
                default_results['Upper bound CI'],
                alpha=1.0,
                color=None,
                facecolor='none',
                zorder=5,
                hatch='||||',
                edgecolor='w',
                linewidth=0)

            # Add labels if the subplot is on the left side of the figure
            if i == 0 or i == 2:
                ax.set_ylabel(r'$R_t^\mathrm{local}$')

            # Add a dotted line for the prior median
            prior_line = ax.axhline(prior_mid,
                                    color='k',
                                    zorder=-20,
                                    ls=':',
                                    lw=2)

            # Add the legend for this epsilon
            ax.legend([(line, shade), ], [r'$ϵ={}$'.format(epsilon), ])

            if i == 0:
                # Add the legend with prior median and default epsilon
                fig.legend([prior_line, (zerorange, zerorangelines, zeroline)],
                           ['Prior median', r'$ϵ={}$'.format(default_epsilon)],
                           bbox_to_anchor=(0.72, 0.67),
                           ncol=2)

            i += 1

    # Use "Jan 01", etc as the date format
    top_ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))

    # Set ticks once per week
    ax.set_xticks(date_times[::7])
    top_ax.set_xticks(data_times[::7])

    # Rotate labels
    plt.xticks(rotation=45, ha='center')
    plt.sca(axs[3])
    plt.xticks(rotation=45, ha='center')
    plt.sca(axs[2])
    plt.xticks(rotation=45, ha='center')

    # Add panel labels
    fig.text(0.025, 0.975, '(a)', fontsize=14)
    fig.text(0.025, 0.63, '(b)', fontsize=14)

    fig.set_size_inches(6, 7)
    fig.set_tight_layout(True)

    if show:
        plt.show()

    return fig
