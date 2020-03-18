import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.legend import Legend
import os
import glob
from utils import get_args, load_timeseries
from sklearn import metrics




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
                                   ((results['Heating Load [MW]'] == 0) | (results['EV Load [MW]'] == 0))].index)

    results = results.drop(results[(results['RGT/LCT'] == 99)].index)
    results = results.drop(results[(results['RGT/LCT'] == 100)].index)

    return results

def plotting_costs_vs_emissions(results):

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(10)

    full_heating_avg = 7715.822081
    baseline_demand_6year_avg = 18655

    colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue']
    cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
    sns.set_palette(sns.xkcd_palette(colors_xkcd))
    sns.set_style("whitegrid")

    for i in range(len(results)):
        lct = results['RGT/LCT'].iloc[i]

        if lct == 100:
            marker = "o"
        elif lct == 99.5:
            marker = 'P'
        elif lct == 99:
            marker = '+'
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


        total_addl_load_frac = round((results['EV Load [MW]'].iloc[i] + results['Heating Load [MW]'].iloc[i])/
                                 (full_heating_avg + args.ev_max_cap) * 100)

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
    patchList_lct.append(ax.scatter([], [], marker="P", edgecolor='k', facecolor='k', label='99.5%'))

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

def plotting_curtailment(results):

    fig, ax = plt.subplots(1,2)
    # fig.set_figwidth(40)
    # fig.set_figheight(20)

    full_heating_avg = 7307.89286986277
    baseline_demand_avg = 18655

    colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue']
    cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
    sns.set_palette(sns.xkcd_palette(colors_xkcd))
    sns.set_style("whitegrid")
    curtailed_energy_GW = np.zeros((len(results)))

    for i in range(len(results)):
        lct = results['RGT/LCT'].iloc[i]

        total_addl_load_frac = round((results['EV Load [MW]'].iloc[i] + results['Heating Load [MW]'].iloc[i]) /
                                     (full_heating_avg + args.ev_max_cap) * 100)
        total_load = baseline_demand_avg + results['EV Load [MW]'].iloc[i] + results['Heating Load [MW]'].iloc[i]

        if lct == 100:
            marker = "o"
        elif lct == 99.5:
            marker = 'P'
        elif lct == 99:
            marker = '+'
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


        curtailed_energy_GW[i] = results['Uncurtailed Renewable Gen. [MW]'].iloc[i] * \
                                    results['Renewable Gen. Curtailment'].iloc[i] /  1000


        scatter = ax[0].scatter(curtailed_energy_GW[i], results['Total LCOE [$/MWh]'].iloc[i],
                   color = point_color, facecolor = facecolor, marker = marker, s = 100)

        if lct == 60 or lct == 80 or lct == 95 or lct == 99.5:
            scatter = ax[1].scatter(results['Uncurtailed Renewable Gen. [MW]'].iloc[i]/1000,
                                 results['Renewable Gen. Curtailment'].iloc[i], color = point_color, facecolor = facecolor,
                                 marker = marker, s = 100)


    best_fit_stats = np.polyfit(curtailed_energy_GW, results['Total LCOE [$/MWh]'], 1,full=True)
    best_fit_curve = np.poly1d(best_fit_stats[0])(curtailed_energy_GW)


    r2_score = metrics.r2_score(results['Total LCOE [$/MWh]'],best_fit_curve)
    print(r2_score)



    ax[0].plot(curtailed_energy_GW, best_fit_curve, color = 'k',
            label = 'Best Fit Curve\n   $r^2$ = {}'.format(np.round(r2_score, 3)))
    ax[0].grid(True)
    ax[1].grid(True)

    ax[0].set_axisbelow(True)
    ax[1].set_axisbelow(True)
    ax[0].legend(loc = 'center left', bbox_to_anchor=(0.00, 0.55))

    patchList_lct = []
    patchList_lct.append(ax[0].scatter([], [], marker="v", edgecolor='k', facecolor='k', label='60%'))
    patchList_lct.append(ax[0].scatter([], [], marker="*", edgecolor='k', facecolor='k', label='70%'))
    patchList_lct.append(ax[0].scatter([], [], marker="X", edgecolor='k', facecolor='k', label='80%'))
    patchList_lct.append(ax[0].scatter([], [], marker="p", edgecolor='k', facecolor='k', label='90%'))
    patchList_lct.append(ax[0].scatter([], [], marker="s", edgecolor='k', facecolor='k', label='95%'))
    patchList_lct.append(ax[0].scatter([], [], marker="d", edgecolor='k', facecolor='k', label='98%'))
    patchList_lct.append(ax[0].scatter([], [], marker="P", edgecolor='k', facecolor='k', label='99.5%'))

    # Colorbar specifications
    cmap_cb = colors.ListedColormap(cmap)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad="6%")
    bounds = np.arange(0, 7)
    boundaries = np.arange(0.5, 6.5, 1)
    tick_names = ['0%', '20%', '40%', '60%', '80%', '100%']
    cb = colorbar.ColorbarBase(cax, cmap=cmap_cb, boundaries=bounds,
                               ticks=boundaries)
    cb.set_ticklabels(tick_names)
    cax.tick_params(length=0)
    cax.set_title('% Elec. of \n Heating and\n Transport', fontsize=16)

    leg_lct = Legend(ax[0], patchList_lct, [l.get_label() for l in patchList_lct], loc='upper left',
                     title="Percent Low-Carbon\nElectricity")
    ax[0].add_artist(leg_lct)
    plt.setp(leg_lct.get_title(), multialignment='center')

    ax[0].set_xlabel('Average Curtailed Renewable Generation [GW]')
    ax[0].set_ylabel('Levelized Cost of Electricity [$/MWh]')
    ax[1].set_xlabel('Average Uncurtailed Renewable Generation [GW]')
    ax[1].set_ylabel('Renewable Generation Curtailment')

    plt.show()


def plotting_peakload(args):


    colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
                   "faded green", 'pastel blue']
    cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
    sns.set_palette(sns.xkcd_palette(colors_xkcd))
    sns.set_style("whitegrid")


    baseline_demand_hourly_mw, heating_hourly, onshore_pot_hourly, offshore_pot_hourly, \
    solar_pot_hourly, fixed_hydro_hourly_mw, flex_hydro_daily_mwh = load_timeseries(args)

    T = len(baseline_demand_hourly_mw)
    ev_charging_array = np.zeros((T, 4))

    ev_charging_rate  = 2

    for j in range(0, int(T / 24) - 1):
        jrange_ev = range(j * 24, j * 24 + args.ev_charging_hours)
        for k in jrange_ev:
            ev_charging_array[args.ev_hours_start + k] = np.array(args.ev_load_dist) * args.ev_max_cap * \
                                                         24 / args.ev_charging_hours


    domain = range(101)

    peak_load_array = np.zeros(len(domain))
    average_load_array = np.zeros(len(domain))

    for i in domain:
        demand = baseline_demand_hourly_mw + i/100 * (heating_hourly + ev_charging_array)
        peak_load_array[i] = np.max(np.sum(demand, axis=1))/1000
        average_load_array[i] = np.mean(np.sum(demand, axis = 1))/1000

    fig, ax = plt.subplots()

    ax.plot(domain, peak_load_array, label = 'Peak Load', color = cmap[0])
    ax.plot(domain, average_load_array, label = 'Average Load', color = cmap[1])
    ax.set_xlabel('Percent Electrification of Heating and Transport')
    ax.set_ylabel('Electricity Load [GW]')
    ax.legend(loc = 'upper left')

    ax.grid(True)

    plt.show()

if __name__ == '__main__':
    args = get_args()


    results_dir = '/Users/terenceconlon/Documents/Columbia - Fall 2019/NYS/model_results/' \
                  'clcpa_results/combined_processed_results'

    results = pd.read_excel(os.path.join(results_dir,
                                         'processed_results_full.xlsx'))
    results = slice_data(results)

    plotting_curtailment(results)


    #