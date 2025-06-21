"""
This script is used to retrieve information from the SUMO scenario.
The script uses the SUMO library to parse the scenario files and extract information such as the maximum road length,
maximum speed, and the mapping of traffic lights with incoming and outgoing edges.
"""

import configparser
import os
from lxml import etree

import sumolib
from utils.file_processing import make_xml, print_xml
import logging

logger = logging.getLogger('scen_retrieve.py')


class SumoScenario:
    def __init__(self, config, log_level='ERROR'):
        self.sumo = None
        self.net_file_path = config.get('net')
        self.rou_file_path = config.get('rou')
        self.cfg_file_path = config.get('cfg')
        self.vtype_file_path = config.get('vtype')
        self.phases_all_tls = {}
        self.lanes = []
        self.tls = []
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def get_lane_length(self, lane_id):
        return self.sumo.lane.getLength(lane_id)

    def max_length(self):
        return max(self.get_lane_length(each) for each in self.specify_lanes())

    def max_length_sumolib(self):
        _, _, _, _, inc_all, out_all = self.node_mapping()
        lane_related = list(set(inc_all+out_all))
        max_length = 0
        for edge in sumolib.output.parse(self.net_file_path, ['edge']):
            if edge.hasChild('lane'):
                for i in range(len(edge['lane'])):
                    if edge['lane'][i].id in lane_related:
                        max_length = float(edge['lane'][i].length) \
                            if float(edge['lane'][i].length) > max_length else max_length
        logger.info(f"max road length: {max_length}")
        return max_length

    def max_speed_lane(self, lane_id):
        return self.sumo.lane.getMaxSpeed(lane_id)

    def max_speed(self):
        return max(self.max_speed_lane(each) for each in self.specify_lanes())

    def specify_lanes(self):
        """collect all lanes to convert index"""
        if not self.lanes:
            for edge in sumolib.output.parse(self.net_file_path, ['edge']):
                if not edge.function:  # function="internal"
                    for lane in edge['lane']:
                        self.lanes.append(lane.id)
        return self.lanes

    def node_mapping(self, tl_chosen=None):
        """
        Map the TL with the incoming edges and the outgoing edges

        Returns
        mapping_inc = dict(tl_id: inc_edges)
        num_inc_edges_max, int, maximum number of inc edges
        mapping_out = dict(tl_id: out_edges)
        num_out_edges_max, int, maximum number of out edges
        -------

        """
        mapping_inc = {}
        num_inc_edges_max = 0
        inc_all = []
        for junction in sumolib.output.parse(self.net_file_path, ['junction']):
            if junction.type == 'traffic_light':
                inc_edges = junction.incLanes.split(' ')
                mapping_inc.update({junction.id: inc_edges})
                inc_all.extend(inc_edges)
                num_inc_edges_max = (len(inc_edges), num_inc_edges_max)[num_inc_edges_max > len(inc_edges)]
        logger.info(f"mapping_inc = dict(tl_id: inc_edges), from north, clockwise: {mapping_inc}")
        logger.info(f"num_inc_edges_max: {num_inc_edges_max}")

        mapping_out = {each: [] for each in mapping_inc.keys()}
        num_out_edges_max = 0
        out_all = []
        for tl_id, inc_edges in mapping_inc.items():
            out_edges = []
            for connection in sumolib.output.parse(self.net_file_path, ['connection']):
                if connection.attr_from + '_' + connection.fromLane in inc_edges:
                    out_edges.append(connection.to + '_' + connection.toLane)
            mapping_out[tl_id] = list(set(out_edges))
            out_all.extend(list(set(out_edges)))
            num_out_edges_max = (len(mapping_out[tl_id]), num_out_edges_max)[
                num_out_edges_max > len(mapping_out[tl_id])]
        logger.info(f"mapping_out = dict(tl_id: out_edges): {mapping_out}")
        logger.info(f"num_out_edges_max: {num_out_edges_max}")
        # return sorted(mapping.items(), key=lambda x: x[0])
        return mapping_inc, num_inc_edges_max, mapping_out, num_out_edges_max, inc_all, out_all

    def get_tls(self):
        if not self.phases_all_tls and not self.tls:
            for tlLogic in sumolib.output.parse(self.net_file_path, ['tlLogic']):
                self.tls.append(tlLogic.id)
        elif self.phases_all_tls and not self.tls:
            self.tls = self.phases_all_tls.keys()
        return self.tls

    def get_phases_all_tls(self):
        """Map traffic light to the list of its signal phases

        Return
        ------
        phases_tl: dict
            {tl_id: [phases]}
        """
        if self.phases_all_tls == {}:
            for tlLogic in sumolib.output.parse(self.net_file_path, ['tlLogic']):
                phases = []
                for each in tlLogic['phase']:
                    phases.append(each.state)
                self.phases_all_tls.update({tlLogic.id: phases})
        logger.info(f"Each traffic light and its signal phases <tl_id: list of phases>: {self.phases_all_tls}")
        return self.phases_all_tls

    def get_max_accel_vtype(self, vtype_id):
        # Todo if no vtype
        for vType in sumolib.output.parse(self.vtype_file_path, ['vType']):
            if vType.id == vtype_id:
                return float(vType.accel)
            return None
        return None

    def generate_sumo_cfg(self):
        sumo_cfg = make_xml('configuration', 'http://sumo.dlr.de/xsd/sumoConfiguration.xsd')

        input_content = etree.Element("input")
        input_content.append(etree.Element("net-file", value=self.net_file_path))
        input_content.append(etree.Element("route-files", value=self.rou_file_path))
        sumo_cfg.append(input_content)

        time_content = etree.Element("time")
        time_content.append(etree.Element("begin", value=repr(0)))
        sumo_cfg.append(time_content)

        file_path = self.net_file_path.split('/')[:-1]
        net_name = self.net_file_path.split('/')[-1].split('.')[0]
        print_xml(sumo_cfg, file_path + net_name + '.sumocfg')
        return file_path + net_name + '.sumocfg'

    def generate_sumo_additional(self):
        # TODO generate sumo add file
        pass

    def green_lanes_per_phase_all_tls(self):
        green_lanes_per_phase_all_tls = {}
        for tl in self.phases_all_tls.keys():
            green_index_per_phase = {}
            green_lanes_per_phase = {}
            phases = self.phases_all_tls[tl]
            for phase in phases:
                if 'G' in phase or 'g' in phase:
                    green_lanes_per_phase.update({phase: []})
                    green_index_per_phase.update({phase: []})
                    for i in range(len(list(phase))):
                        if list(phase)[i] == 'G' or list(phase)[i] == 'g':
                            green_index_per_phase[phase].append(i)
            for connection in sumolib.output.parse(self.net_file_path, ['connection']):
                if connection.tl:
                    if connection.tl == tl:
                        for phase, green_index in green_index_per_phase.items():
                            if int(connection.linkIndex) in green_index:  # have duplicate lanes
                                green_lanes_per_phase[phase].append(connection.attr_from + '_' + connection.fromLane)
            # delete duplicate lane
            for phase, lanes in green_lanes_per_phase.items():
                green_lanes_per_phase[phase] = list(set(lanes))

            green_lanes_per_phase_all_tls.update({tl: green_lanes_per_phase})
        return green_lanes_per_phase_all_tls


if __name__ == '__main__':
    config_file = 'xTL.ini'
    config = configparser.ConfigParser()
    config.read(os.path.join('./../exp_configs', config_file))

    JuncRed = SumoScenario(config['SCEN_CONFIG'], log_level="INFO")
    JuncRed.node_mapping()
    JuncRed.get_phases_all_tls()
    JuncRed.max_length_sumolib()
