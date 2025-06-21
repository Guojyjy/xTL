"""
This script is used to run non-DRL simulations using SUMO.
It takes in a sumo configuration file and runs the simulation for a specified number of runs.
The output files can be generated if specified.
The script uses the TraCI interface to communicate with SUMO.

"""
import argparse
import sys
import os
import traci
import time
from utils.file_processing import ensure_dir


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a non-DRL simulation.",
        epilog="python simulate.py PATH_sumo-cfg --num_runs INT --no_render")

    # required input parameters
    parser.add_argument(
        '--path_config', type=str,
        default="/home/gjy/coach/scenario/Jun1_3turn/90red/90red_3turn.sumocfg",
        help='Path of the sumo configuration file, in scenario.')

    # optional input parameters
    parser.add_argument(
        '--num_runs', type=int, default=10,
        help='Number of simulations to run. Defaults to 1.')
    parser.add_argument(
        '--render',
        action='store_false',
        help='Specifies whether to run the GUI simulation during runtime.')
    parser.add_argument(
        '--no_output',
        action='store_false',
        help='Specifies whether to generate output files (tripinfo, emission, queue, ssm) from the '
             'simulation.')
    parser.add_argument(
        '--output_scenario', type=str,
        default="Jun1_90_3turnRed_Fixed",
        help='Name the directory to store output file, in scenario.')

    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])
    if flags.path_config:
        path_cfg = flags.path_config
        num_runs = int(flags.num_runs)
        render = flags.render
        no_output = flags.no_output
        output_scenario = flags.output_scenario

        for i in range(num_runs):
            sumo_binary = "sumo-gui" if render else "sumo"
            sumo_cmd = [sumo_binary, "-c", path_cfg,
                        "--step-length", "5", # step-length in seconds
                        # "--no-warnings",
                        "--scale", "1",
                        "--tripinfo-output.write-unfinished",
                        "--random",
                        "-S",  # Start the simulation after loading
                        "-Q",  # Quits the GUI when the simulation stops
                        ]

            if no_output:
                scen = path_cfg.split('.sumo')[0].split('/')[-1]
                output_path = os.path.join(path_cfg.split('.sumo')[0].split('scenario')[0], f"output/{output_scenario}/")
                ensure_dir(output_path)
                output_filetypes = ["emission-output", "tripinfo-output", "queue-output"]
                for each in output_filetypes:
                    file_name = os.path.join(output_path,
                                             f"{i}-{each.split('-')[0]}.xml")
                    sumo_cmd.extend([f"--{each}", file_name])

                # # obtain safety info
                file_name = os.path.join(os.getcwd().split('scenario')[0], output_path, f"{i}-ssm.xml")

                sumo_cmd.extend(["--device.ssm.deterministic", "true"])
                sumo_cmd.extend(["--device.ssm.measures", "TTC DRAC PET BR SGAP TGAP"])
                sumo_cmd.extend(["--device.ssm.thresholds", "3.0 3.0 2.0 0.0 0.2 0.5"])
                sumo_cmd.extend(["--device.ssm.range", "50.0"])
                sumo_cmd.extend(["--device.ssm.extratime", "5.0"])
                sumo_cmd.extend(["--device.ssm.trajectories", "true"])
                sumo_cmd.extend(["--device.ssm.geo", "false"])
                sumo_cmd.extend(["--device.ssm.file", file_name])

            traci.start(sumo_cmd, numRetries=100, label=str(time.time()))
            # for _ in range(360): # run for 360 * step-length seconds
            while traci.simulation.getMinExpectedNumber() > 0: # run until there are no more vehicles
                traci.simulationStep()
            traci.close()
    else:
        raise ValueError("Unable to find necessary options: --path_config.")
