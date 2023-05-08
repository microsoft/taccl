# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import numpy as np
import os
from taccl.routing import TACCLRouting
from taccl.heuristic_ordering import HeuristicOrderer
from taccl.scheduler import TACCLScheduler
from taccl.reduce_scheduler import TACCLRevScheduler
from .known_collectives import KnownCollectives
from .known_topologies import KnownTopologies
from .common import *

def optimize_comm_sketch(topology, route_sketch, collective, distribute_over_links=False):
    path_encoder = TACCLRouting(topology, route_sketch, collective)
    orderer = HeuristicOrderer(topology, route_sketch, collective)
    scheduler = TACCLScheduler(topology, route_sketch, collective)

    chunk_send, time_send, chunk_recv, time_recv = path_encoder.optimize(distribute_over_links)
    time_recv, chunk_recv, switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, _ = orderer.perform_ordering(
        chunk_send, time_send, chunk_recv, time_recv
    )
    cont_algo = scheduler.optimize(chunk_recv, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send)
    return cont_algo


def check_heur_comm_sketch(topology, route_sketch, collective, ts_heur):
    path_encoder = TACCLRouting(topology, route_sketch, collective)
    orderer = HeuristicOrderer(topology, route_sketch, collective)
    scheduler = TACCLScheduler(topology, route_sketch, collective)

    chunk_send, time_send, chunk_recv, time_recv = path_encoder.check_heuristic(ts_heur)
    time_recv, chunk_recv, switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, _ = orderer.perform_ordering(
        chunk_send, time_send, chunk_recv, time_recv
    )
    cont_algo = scheduler.optimize(chunk_recv, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send)
    return cont_algo

def get_send_dict_base(ts=""):
    assert len(ts)
    return np.load(f"send_dict_{ts}.npy", allow_pickle=True).item()

def process_dict(send_dict_base, topology, collective):
    C = collective.num_chunks
    R = collective.num_nodes
    L = topology.L

    time_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
    chunk_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
    time_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
    chunk_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]

    for t in send_dict_base:
        for (c,src,r,t_,l) in send_dict_base[t]:
            chunk_send[src][r][l].append(c)
            time_send[src][r][l].append(t_)
            chunk_recv[r][src][l].append(c)
            time_recv[r][src][l].append(t_ + topology.get_invbw(src,r))
    return chunk_send, time_send, chunk_recv, time_recv

def optimize_reduction(reduce_coll, topology, route_sketch, collective, ts, prefer_local_reduce_first=False):
    orderer = HeuristicOrderer(topology, route_sketch, collective, reverse=True)
    scheduler = TACCLRevScheduler(topology, route_sketch, collective)

    send_dict_base = get_send_dict_base(ts)
    chunk_send, time_send, chunk_recv, time_recv = process_dict(send_dict_base, topology, collective)

    # heuristic = 12 in routesketch will reverse the chunk order
    time_recv, chunk_order,switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, paths = orderer.perform_ordering(chunk_send, time_send, chunk_recv, time_recv)
    for r in range(collective.num_nodes):
        for ll in range(len(switch_chunk_recv[r])):
            print("new_swt_recv: ", r, ll, switch_chunk_recv[r][ll])
    for r1 in range(len(chunk_order)):
        for r2 in range(len(chunk_order[r1])):
            for l in range(len(chunk_order[r1][r2])):
                if len(chunk_order[r1][r2][l]):
                    print("old_send_order", r1, r2, chunk_recv[r2][r1][l])
                    print("new_send_order", r2, r1, chunk_order[r1][r2][l])

    ordered_send_dict_reverse = scheduler.optimize_reversed(chunk_order, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send, paths)
    np.save(f'send_dict_redscat_{ts}.npy', ordered_send_dict_reverse)

    cont_algo = scheduler.build_allreduce(reduce_coll,ordered_send_dict_reverse, send_dict_base, ts)

    return cont_algo

def make_handle_solve_comm_sketch(cmd_parsers):
    name = 'solve'
    cmd = cmd_parsers.add_parser(name)
    topologies = KnownTopologies(cmd)
    collectives = KnownCollectives(cmd)
    validate_output_args, output_handler = add_output_sccl_objects(cmd)
    # cmd.add_argument('--topo-file', type=argparse.FileType('r'))
    cmd.add_argument('--sketch-file', type=argparse.FileType('r'))
    # cmd.add_argument('--topo-name', type=str)
    cmd.add_argument('--ts-heur', type=int, default="-1")
    def handle(args, command):
        if command != name:
            return False

        validate_output_args(args)
        node_topology = topologies.create(args)
        topology, route_sketch = parse_and_get_topo(node_topology, args.sketch_file)
        collective = collectives.create(args, topology.num_nodes()).chunk_up(route_sketch.hyperparameters.chunkup)
        ts_heur = args.ts_heur
        if ts_heur == -1:
            algo = optimize_comm_sketch(topology, route_sketch, collective)
        else:
            algo = check_heur_comm_sketch(topology, route_sketch, collective, ts_heur)
        output_handler(args, algo, algo.name + "_taccl")
        return True
    
    return handle

def make_handle_combine_comm_sketch(cmd_parsers):
    name = 'combine'
    cmd = cmd_parsers.add_parser(name)
    topologies = KnownTopologies(cmd)
    collectives = KnownCollectives(cmd)
    validate_output_args, output_handler = add_output_sccl_objects(cmd)
    cmd.add_argument('--sketch-file', type=str, default=None)
    cmd.add_argument('--ts', type=str, help='timestamp of send_dict for Allgather')
    cmd.add_argument('--prefer-local-reduce-first', action='store_true', help='should prefer reducing a chunk locally first if it is the same either way')
    def handle(args, command):
        if command != name:
            return False
        if args.sketch_file is None:
            cmd_parsers.error('Must specify sketch file')

        assert os.path.isfile(args.sketch_file), "sketch file does not exist"
        sketch_file = open(args.sketch_file, 'r')

        validate_output_args(args)
        node_topology = topologies.create(args)
        topology, route_sketch = parse_and_get_topo(node_topology, sketch_file, reduce=True)
        collective = collectives.create(args, topology.num_nodes()).chunk_up(route_sketch.hyperparameters.chunkup)

        import copy
        new_args = copy.deepcopy(args)
        # new_args.collective = 'Allgather'
        # new_args.collective = 'ReduceScatter'
        new_args.collective = 'Allreduce'
        allreduce_coll = collectives.create(new_args, topology.num_nodes()).chunk_up(route_sketch.hyperparameters.chunkup)
        algo = optimize_reduction(allreduce_coll, topology, route_sketch, collective, args.ts, args.prefer_local_reduce_first)
        output_handler(args, algo, algo.name + "_taccl")
        return True

    return handle