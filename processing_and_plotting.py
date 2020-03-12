import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.legend import Legend
from matplotlib import rc
import os
import glob
from utils import get_args, clcpa_results_compiling


def slice_data(results):

    ## Only plot data without nuclear
    results = results[results['Nuclear Binary'] == 0]
    ## Only plot results without additional HQ imports
    results = results[results['HQ-CH Addl. Cap.'] == 0]
    ## Limit data to only no-H2
    results = results[results['H2 Binary'] == 0]
    # Drop individual additional load cases when RGT > 80
    results = results.drop(results[(results['RGT/LCT'] <= 50)].index)
    #
    results = results.drop(results[(results['RGT/LCT'] > 95) &
                                   ((results['Heating Load'] == 0) | (results['EV Load'] == 0))].index)

    return results

def plotting(results):

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(10)

    colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue']
    cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
    sns.set_palette(sns.xkcd_palette(colors_xkcd))
    sns.set_style("whitegrid")

    for i in range(len(results)):
        lct = results['RGT/LCT'].iloc[i]

        if lct == 100:
            marker = "o"
        elif lct == 98:
            marker = "d"
        elif lct == 95:
            marker = "s"
        elif lct == 90:
            marker = 'p'
        elif lct == 80:
            marker = 'X'
        elif lct == 70:
            marker = '*'
        elif lct == 60:
            marker = 'v'
        elif lct == 50:
            marker = 'd'
        elif lct == 0:
            marker = 'P'

        total_addl_load_frac = round((results['EV Load'].iloc[i] + results['Heating Load'].iloc[i])/
                                 (args.heating_max_cap + args.ev_max_cap) * 100)

        if total_addl_load_frac == 0:
            point_color = cmap[0]
        elif total_addl_load_frac == 20:
            point_color = cmap[1]
        elif total_addl_load_frac == 40:
            point_color = cmap[2]
        elif total_addl_load_frac == 60:
            point_color = cmap[3]
        elif total_addl_load_frac == 80:
            point_color = cmap[4]
        elif total_addl_load_frac == 100:
            point_color = cmap[5]

        facecolor = point_color

        scatter = ax.scatter(results['GHG Reduction'].iloc[i] * 100, results['Total LCOE [$/MWh]'].iloc[i],
                             color=point_color, facecolor=facecolor, marker=marker, s=120)

    # Set legend for LCTs
    patchList_lct = []
    patchList_lct.append(ax.scatter([], [], marker="v", edgecolor='k', facecolor='k', label='60%'))
    patchList_lct.append(ax.scatter([], [], marker="*", edgecolor='k', facecolor='k', label='70%'))
    patchList_lct.append(ax.scatter([], [], marker="X", edgecolor='k', facecolor='k', label='80%'))
    patchList_lct.append(ax.scatter([], [], marker="p", edgecolor='k', facecolor='k', label='90%'))
    patchList_lct.append(ax.scatter([], [], marker="s", edgecolor='k', facecolor='k', label='95%'))
    patchList_lct.append(ax.scatter([], [], marker="d", edgecolor='k', facecolor='k', label='98%'))
    patchList_lct.append(ax.scatter([], [], marker="o", edgecolor='k', facecolor='k', label='100%'))

    # Colorbar specifications
    cmap_cb = colors.ListedColormap(cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="6%")
    bounds = np.arange(0, 7)
    boundaries = np.arange(0.5, 6.5, 1)
    tick_names = ['0%', '20%', '40%', '60%', '80%', '100%']
    cb = colorbar.ColorbarBase(cax, cmap=cmap_cb, boundaries=bounds,
                               ticks=boundaries)
    cb.set_ticklabels(tick_names)
    cax.tick_params(length=0)
    cax.set_title('% Elec. of \n Heating and\n Transport', fontsize=16)

    leg_lct = Legend(ax, patchList_lct, [l.get_label() for l in patchList_lct], loc='upper left',
                     title="Percent Low-Carbon\nElectricity")
    ax.add_artist(leg_lct)
    plt.setp(leg_lct.get_title(), multialignment='center')


    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.set_xlabel('Percent Reduction in NYS GHG Emissions (cf. 1990 Levels)')
    ax.set_ylabel('Levelized Cost of Electricity [$/MWh]')

    ax.grid(True)
    ax.set_axisbelow(True)

    return fig, ax


if __name__ == '__main__':
    args = get_args()
    compile_results = False

    if compile_results:
        clcpa_results_compiling(args)

    results = pd.read_excel(os.path.join(args.results_dir, 'clcpa_results', 'processed_results_compiled.xlsx'))
    results = slice_data(results)

    fig, ax = plotting(results)

    plt.show()