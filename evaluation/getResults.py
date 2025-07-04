import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib


def choose_files(path, scenario, avg_num, recover_bool, safety):
    """
    Choose the corresponding output files for multiple simulation episodes, tripinfo files included as default

    Parameters
    ----------
    path: str, path to scenario directory
    scenario: str, scenario name
    avg_num: int, how many episodes to get the average, default is 8
    recover_bool: bool, whether to calibrate output file format

    Returns
    -------
    file_dict: dict, {file_type: [file_name*avg_num]}
    """
    file_num = 0
    if safety:
        related_file_list = ['-ssm.xml']  # except for tripinfo file, such as ['-emission.xml', '-queue.xml', '_emission.csv']
        file_dict = {'-tripinfo.xml': [], '-ssm.xml': []}
    else:
        related_file_list = []
        file_dict = {'-tripinfo.xml': []}
    scenarios_list = os.listdir(path)
    if scenario in scenarios_list:
        dir_path = os.path.join(path, scenario)
        for file in os.listdir(dir_path):
            if file.endswith('tripinfo.xml'):
                if recover_bool:
                    recover_xml(dir_path + '/' + file)
                file_dict['-tripinfo.xml'].append(file)
                file_num += 1

                file_suffix = file.split('-tripinfo')[0]
                for each in related_file_list:
                    file_name = file_suffix + each
                    if recover_bool:
                        recover_xml(dir_path + '/' + file)
                    file_dict[each].append(file_name)
            if file_num == avg_num:
                break
    print('Evaluated files:', file_dict['-tripinfo.xml'])
    return file_dict


def recover_xml(file_path):
    """ Recovers incomplete output xml files written during the simulation,
    closing tag element is probably missing when simulation terminates """
    cmd = "xmllint --valid " + file_path + " --recover --output " + file_path
    print()
    print('-----')
    print('Recover the xml file')
    os.system(cmd)


def get_main_metric(tripinfo_files, path, speed_limit):
    print()
    print('---------')
    print(path.split('/')[-1])
    fuel = [0] * len(tripinfo_files)
    CO2 = [0] * len(tripinfo_files)
    duration = [0] * len(tripinfo_files)
    all_trip_length = []
    finished_trip_num = []
    tt_veh = {}  # {veh_id: [travel_time_i]}
    for i in range(0, len(tripinfo_files)):
        trip_length = {}  # {trip length(m): vehicle count}
        try:
            print(tripinfo_files[i])
            for trip in sumolib.output.parse(path + '/' + tripinfo_files[i], ['tripinfo']):
                # sum of travel time
                duration[i] += float(trip.duration) if trip.duration else 0  # last vehicle not written well
                if trip.hasChild('emissions'):  # last vehicle not written well
                    fuel[i] += float(trip['emissions'][0].fuel_abs) if trip['emissions'][0].fuel_abs else 0
                    CO2[i] += float(trip['emissions'][0].CO2_abs) if trip['emissions'][0].CO2_abs else 0
                triplength = float(trip.routeLength) if trip.routeLength else 0
                # count different trip length
                if trip_length.get(triplength):
                    trip_length[triplength] += 1
                else:
                    trip_length.update({triplength: 1})

                # travel time for individual vehicles
                if trip.id not in tt_veh.keys():
                    tt_veh.update({trip.id: [float(trip.duration) if trip.duration else 0]})
                else:
                    tt_veh[trip.id].append(float(trip.duration))
        except ET.ParseError as e:
            print("Error:", e)
            print(f"The XML file: {tripinfo_files[i]} has an incomplete or incorrect format.")

        all_trip_length.append(sum(triplength * num for triplength, num in trip_length.items()))
        finished_trip_num.append(sum(num for num in trip_length.values()))  # only finished trip in tripinfo file

    # get average value and convert unit
    avg_all_trip_length = np.mean(all_trip_length)
    avg_finished_trip_num = np.mean(finished_trip_num)
    print('Average finished trip number in this scenario: ', avg_finished_trip_num)
    print('Average vehicle trip length (m): ', avg_all_trip_length / avg_finished_trip_num)
    fuel = np.mean(fuel / (avg_all_trip_length / 100))  # ml/m -> l/100km // mg/m -> g/100km
    CO2 = np.mean(CO2 / avg_all_trip_length)  # mg/m -> g/km
    delay = np.mean(duration / avg_finished_trip_num) - avg_all_trip_length / speed_limit / avg_finished_trip_num
    duration_std = np.std(duration / avg_finished_trip_num)
    duration = np.mean(duration / avg_finished_trip_num)

    return fuel, CO2, duration, duration_std, delay, tt_veh


def get_safety_metric(ssm_files, path):
    TTC = [0] * len(ssm_files)
    for i in range(0, len(ssm_files)):
        TTC_count = 0
        try:
            for conflict in sumolib.output.parse(path + '/' + ssm_files[i], ['conflict']):
                try:
                    TTCSpan = (conflict['TTCSpan'][0].values).split(' ')
                    for each in TTCSpan:
                        if each == 'NA':
                            pass
                        elif float(each) <= 3.00:
                            TTC_count += 1
                except:
                    pass
        except ET.ParseError as e:
            print("Error:", e)
            print(f"The XML file: {ssm_files[i]} has an incomplete or incorrect format.")
        TTC[i] = TTC_count
    print("TTC:", TTC)
    print('Average TTC number < 3.0 sec: %f' % (np.mean(TTC)))
    return np.mean(TTC)

def have_df_metric(scen_file_dict, path_dir, speed_limit, scenarios_legend, safety):
    avg_fuel = []
    avg_CO2 = []
    avg_dura = []
    avg_duraStd = []
    avg_delay = []
    avg_ttc = []

    travelt_all = []
    traveltime_df = pd.DataFrame()
    scen_tag = []

    for scenario in scen_file_dict.keys():
        fuel, CO2, duration, durationStd, delay, travelt = get_main_metric(scen_file_dict[scenario]['-tripinfo.xml'],
                                                              path_dir + '/' + scenario, speed_limit)
        avg_fuel.append(fuel)
        avg_CO2.append(CO2)
        avg_dura.append(duration)
        avg_duraStd.append(durationStd)
        avg_delay.append(delay)

        this_traveltime_df = pd.DataFrame()
        traveltime_avg_each_vehicle = []
        for tt_list in travelt.values():  # for each vehicle, ValueError: Length of traveltime_df values does not match
            traveltime_avg_each_vehicle.append(np.mean(tt_list))

        this_traveltime_df.insert(len(traveltime_df.columns), scenario, traveltime_avg_each_vehicle)
        print("Statistics of travel time (sec)")
        print(this_traveltime_df.describe())

        if safety:
            ttc = get_safety_metric(scen_file_dict[scenario]['-ssm.xml'], path_dir + '/' + scenario)
            avg_ttc.append(ttc)

    dataframe = pd.DataFrame(
        {'Scenario': scen_file_dict.keys(), 'Fuel consumption l/100km': avg_fuel, 'CO2 emission g/1km': avg_CO2,
         'Average travel time': avg_dura, 'Std travel time': avg_duraStd, 'Delay': avg_delay})

    if safety:
        dataframe.insert(dataframe.shape[1], 'TTC', avg_ttc)

    print()
    print("Main traffic statistics:")
    print(dataframe)
    return travelt_all, scen_tag


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when evaluating an experiment result.",
        epilog="python outputFilesProcessing.py")

    # necessary
    parser.add_argument(
        '--scen', nargs='+',
        help='The list of several scenario names.')

    # optional
    parser.add_argument(
        '--output_dir', type=str, default="../output/",
        help='The directory of output files contains multiple scenarios.')

    parser.add_argument(
        '--scen_legend', nargs='+',
        help='The list of several scenario names in figure legend.')

    parser.add_argument(
        '--speed_limit', type=int, default=15,
        help='Speed limit.')

    parser.add_argument(
        '--avg_num', type=int, default=10,
        help='How many scenarios to get the average result.')

    parser.add_argument(
        '--recover', type=bool, default=False,
        help='About xml.etree.ElementTree.ParseError, set to True')

    parser.add_argument(
        '--no_safety', action='store_false',
        help='Whether to include traffic safety statistics (i.e., TTC)')


    return parser.parse_known_args(args)[0]


def main(args):
    flags = parse_args(args)
    if flags.scen:
        path_dir = flags.output_dir
        scenarios_list = flags.scen
        scenarios_legend = flags.scen_legend
        speed_limit = flags.speed_limit
        avg_num = flags.avg_num
        recover_bool = flags.recover
        safety = flags.no_safety
        # choose some scenarios for assessment
        scen_file_dict = {}
        for scenario in scenarios_list:
            scen_file_dict.update({scenario: choose_files(path_dir, scenario, avg_num, recover_bool, safety)})
        # show the main metric results, CO2 emissions, fuel consumption, travel time, and delay
        have_df_metric(scen_file_dict, path_dir, speed_limit, scenarios_legend, safety)

    else:
        raise ValueError("Unable to find necessary options: --scen.")


if __name__ == '__main__':
    main(sys.argv[1:])
