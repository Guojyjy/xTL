import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def list_to_float(list_):
    try:
        return [float(item) for item in list_]
    except ValueError as e:
        print(f"Error converting list to float: {e}")
        return []


def read_file(file, legend_tag):
    interation = []
    avg = []
    tag_this = []
    print(f"Reading file: {file}")
    data = pd.read_csv(file)
    tt = list_to_float(list(data['training_iteration']))
    interation += tt * 3

    avg += list_to_float(list(data['episode_reward_mean']))
    avg_values = np.array(list_to_float(list(data['episode_reward_mean'])))
    # add policy_reward_std in python3.7/site-packages/ray/rllib/evaluation/metrics.py
    std_values = np.array(list_to_float(list(data['episode_reward_std'])))
    avg += list(avg_values + std_values)
    avg += list(avg_values - std_values)
    tag_this += [legend_tag] * len(avg_values) * 3
    return interation, avg, tag_this


def draw_plot(interation_num, avg_reward, tagg):

    dataframe = pd.DataFrame({'Training iteration': interation_num, 'Average reward': avg_reward, ' ': tagg})
    print(dataframe)

    fig, ax = plt.subplots()
    sns.lineplot(x=dataframe["Training iteration"], y=dataframe["Average reward"], ax=ax, hue=dataframe[" "])
    sns.color_palette('bright')
    plt.legend(title='', loc='lower right', fontsize='13')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Average Episode Reward", fontsize=10)
    plt.grid()
    plt.show()


def isolated():
    # xTL
    scenarios_list = [
            'xTL/IntelliLight',
            'xTL/ImageTL11122',
            'xTL/xTL']
    scenarios_legend = ['IntelliLight', 'xTL(image)', 'xTL']
    return scenarios_list, scenarios_legend


def main():
    path = '../ray_results/'
    prog = '/progress.csv'
    scen_list, scen_legend = isolated()

    interation_num = []
    avg_reward = []  # include std range
    tagg = []

    for each in scen_list:
        interation, avg, tag_this = read_file(path+each+prog, scen_legend[scen_list.index(each)])
        interation_num += interation
        avg_reward += avg
        tagg += tag_this

    draw_plot(interation_num, avg_reward, tagg)


if __name__ == '__main__':
    main()
