# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from taccl.serialization import *
from taccl.instance import *
from taccl.topologies import TACCLTopology, IntraNode, IntraNode_Switch, InterNode, InterNode_Relay, MultiNode, Symmetry, HyperParameter, RouteSketch, NodeTopology

import json
from pathlib import Path
import sys
import re
from fractions import Fraction
from collections import defaultdict

def _legalize_sccl_name(name):
    name = name.replace('(', '.')
    name = name.replace('=', '')
    name = name.replace(',', '.')
    name = name.replace(')', '')
    return name

def name_sccl_object(name, ending='sccl.json'):
    return f'{_legalize_sccl_name(name)}.{ending}'

def _validate_output_directory(directory):
    if not directory.exists():
        print('error: output directory does not exists', file=sys.stderr)
        exit(1)
    if not directory.is_dir():
        print('error: output path is not a directory', file=sys.stderr)
        exit(1)

def _handle_write_to_directory(directory, force, get_contents, preferred_file_name):
    output_file = directory / preferred_file_name
    if output_file.exists():
        if output_file.is_dir():
            print(f'error: output path is a directory', file=sys.stderr)
            exit(1)
        if force:
            print(f'Overwriting {output_file}')
        else:
            print(f'file already exists, use -f/--force to overwrite {output_file}', file=sys.stderr)
            return False
    with output_file.open('w') as f:
        f.write(get_contents())
    print(f'Wrote to {output_file}')
    return True

def add_output_file(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-o', '--output', type=Path, help='file to write synthesized algorithm to', metavar='FILE')
    group.add_argument('-d', '--directory', type=Path, default=Path(), help='directory to write the synthesized algorithm to', metavar='DIR')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing files')
    parser.add_argument('--no-save', action='store_true', help='do not save to file')

    def validate_args(args):
        if args.output != None:
            if args.output.is_dir():
                print(f'error: output path is a directory, did you mean to use -d?', file=sys.stderr)
                exit(1)
        if args.directory != None:
            _validate_output_directory(args.directory)

    def handle(args, get_contents, preferred_file_name):
        if args.no_save:
            return False
        if args.output != None:
            if args.output.exists() and not args.force:
                print(f'file already exists, use -f/--force to overwrite {args.output}', file=sys.stderr)
                return False
            with args.output.open('w') as f:
                f.write(get_contents())
            print(f'Wrote to {args.output}')
        else:
            return _handle_write_to_directory(args.directory, args.force, get_contents, preferred_file_name)
        return True

    return validate_args, handle

def add_output_algorithm(parser):
    validate_args, handle_file = add_output_file(parser)

    def handle(args, algorithm):
        if algorithm == None:
            return # Strategies/distributors have their specific failure prints

        handled = handle_file(args, lambda: SCCLEncoder().encode(algorithm), name_sccl_object(algorithm.name))
        if not handled:
            print(f'\n{algorithm.name} algorithm:')
            print(algorithm)

    return validate_args, handle

def add_output_topology(parser):
    validate_args, handle_file = add_output_file(parser)

    def handle(args, topology):
        handled = handle_file(args, lambda: SCCLEncoder().encode(topology), name_sccl_object(topology.name))

    return validate_args, handle

def add_output_sccl_objects(parser):
    parser.add_argument('-d', '--directory', type=Path, default=Path(), help='directory to write outputs to', metavar='DIR')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing files')
    parser.add_argument('--no-save', action='store_true', help='do not save to file')

    def validate_args(args):
        _validate_output_directory(args.directory)

    def handle(args, sccl_object, name):
        if not args.no_save:
            _handle_write_to_directory(args.directory, args.force, lambda: SCCLEncoder().encode(sccl_object), name_sccl_object(name))
    
    return validate_args, handle

def add_input_algorithm(parser, multiple=False, name='algorithm'):
    parser.add_argument(name, type=Path, nargs='+' if multiple else 1, help=f'algorithm to operate on')

    def read_algorithm(args):
        algos = []
        for input_file in vars(args)[name]:
            if not input_file.exists():
                print(f'error: input file not found: {input_file}', file=sys.stderr)
                exit(1)

            algo = load_sccl_object(input_file)
            algos.append(algo)
        if multiple:
            return algos
        else:
            return algos[0]

    return read_algorithm

def add_instance(parser, take_steps=True, take_rounds=True, take_chunks=True):
    if take_steps:
        parser.add_argument('-s', '--steps', type=int, required=True)
    if take_rounds:
        parser.add_argument('-r', '--rounds', type=int, default=None, metavar='N')
    if take_chunks:
        parser.add_argument('-c', '--chunks', type=int, default=1, metavar='N')
    parser.add_argument('--pipeline', type=int, default=None, metavar='N')
    parser.add_argument('--extra-memory', type=int, default=None, metavar='N')
    parser.add_argument('--allow-exchange', action='store_true')

    def handle(args):
        if take_rounds:
            if args.rounds != None:
                if args.rounds < args.steps:
                    parser.error(f'error: rounds cannot be less than steps ({args.rounds} < {args.steps})')
                extra_rounds = args.rounds - args.steps
            else:
                extra_rounds = 0
        return Instance(
            steps=args.steps if take_steps else None,
            extra_rounds=extra_rounds if take_rounds else 0,
            chunks=args.chunks if take_chunks else 1,
            pipeline=args.pipeline,
            extra_memory=args.extra_memory,
            allow_exchange=args.allow_exchange)

    return handle

def parse_fraction(value):
    try:
        return int(value)
    except ValueError:
        m = re.fullmatch('(.+)/(.+)', value)
        if m == None:
            raise ValueError('value must be in format "<numerator>/<denominator>"')
        numerator = int(m.group(1))
        denominator = int(m.group(2))
        return Fraction(numerator, denominator)

def make_cmd_category(cmd_parsers, name, title, handler_funcs):
    cmd = cmd_parsers.add_parser(name)
    category_parsers = cmd.add_subparsers(title=title, dest=title)
    category_parsers.required = True
    
    handlers = []
    for func in handler_funcs:
        handlers.append(func(category_parsers))
    
    def handle(args, command):
        if command != name:
            return False
        
        for handler in handlers:
            if handler(args, vars(args)[title]):
                return True
    
    return handle

def _multiply_link_matrix(links, factor_matrix):
    new_links = [[links[dst][src] * factor_matrix[dst][src] for src in range(len(links))] for dst in range(len(links[0]))]
    return new_links

def _div_beta_add_alpha(alpha, betas, factor_matrix):
    new_betas = [[ int(betas[dst][src] / factor_matrix[dst][src]) for src in range(len(betas))] for dst in range(len(betas[0]))]
    new_invbws = [[ int(alpha + betas[dst][src] / factor_matrix[dst][src]) for src in range(len(betas))] for dst in range(len(betas[0]))]
    return new_betas, new_invbws

def _filter_links(links, conn):
    new_links = [
        [
            links[dst][src]
            if src in conn and dst in conn[src]
            else 0
            for src in range(len(links))
        ] for dst in range(len(links[0]))
    ]
    return new_links

def _filter_invbws(invbws, conn):
    new_invbws = [
        [
            invbws[dst][src] * len(conn[src])
            if src in conn and dst in conn[src]
            else 0
            for src in range(len(invbws))
        ] for dst in range(len(invbws[0]))
    ]
    return new_invbws



def parse_and_get_topo(node_topology: NodeTopology, comm_sketch_file, reduce=False):
    cs_json = json.load(comm_sketch_file)
    copies = cs_json["nnodes"]
    ngpus_per_node = len(node_topology.links)
    switches = []

    if cs_json["intranode_sketch"]["strategy"] == "switch":
        assert len(cs_json["intranode_sketch"]["switches"]) == len(cs_json["intranode_sketch"]["switch_hyperedge_strategy"])
        intranode_sketch = IntraNode_Switch(
            cs_json["intranode_sketch"]["strategy"],
            cs_json["intranode_sketch"]["switches"],
            cs_json["intranode_sketch"]["switch_hyperedge_strategy"]
        )
        switches = cs_json["intranode_sketch"]["switches"]

        intra_node_split = [[1 for _ in range(ngpus_per_node)] for _ in range(ngpus_per_node)]
        # Assert that the interesection of any two sets of switches is either empty or equal to the set of switches
        # This is required for correctly deriving the way to split the bandwidth in the node and update the links
        # Get the number of disjoint sets of switches and update the intra_node_split matrix
        added = []
        intersections = {}
        for i in range(len(switches)):
            if (i not in added):
                num_same = 1
                for j in range(i+1, len(switches)):
                    intersection = list(set(switches[i]) & set(switches[j]))
                    assert len(intersection) == 0 or (len(intersection) == len(switches[i]) and len(intersection) == len(switches[j]))
                    if len(intersection):
                        num_same += 1
                        added.append(j)
                for gpu_i in switches[i]:
                    for gpu_j in switches[i]:
                        if (gpu_i != gpu_j):
                            intra_node_split[gpu_i][gpu_j] = num_same
                            intra_node_split[gpu_j][gpu_i] = num_same
                added.append(i)
        for row in intra_node_split:
            print(row)
        new_switches = [[n for n in switches[i]] for i in added]
        switches = new_switches
        # Update the links and invbws
        new_links = _multiply_link_matrix(node_topology.links, intra_node_split)
        new_betas, new_invbws = _div_beta_add_alpha(node_topology.alpha, node_topology.betas, intra_node_split)
        node_topology.links = new_links
        node_topology.betas = new_betas
        node_topology.invbws = new_invbws
    elif cs_json["intranode_sketch"]["strategy"] == "maxmin" or cs_json["intranode_sketch"]["strategy"] == "minmax":
        intranode_sketch = IntraNode(cs_json["intranode_sketch"]["strategy"])
    elif cs_json["intranode_sketch"]["strategy"] == "none":
        intranode_sketch = IntraNode("none")
    else:
        assert False, "No such intranode strategy available"

    if copies > 1:
        nics_per_node = node_topology.nics_per_node
        assert "internode_sketch" in cs_json
        assert cs_json["internode_sketch"]["strategy"] == "relay"
        assert "internode_conn" in cs_json["internode_sketch"]
        internode_conn = cs_json["internode_sketch"]["internode_conn"]
        if not isinstance(internode_conn, dict):
            assert isinstance(internode_conn, str)
            conns = defaultdict(list)
            if internode_conn == "fully-connected":
                for i in range(ngpus_per_node):
                    for j in range(ngpus_per_node):
                        conns[i].append(j)
            elif internode_conn == "direct-map":
                for i in range(ngpus_per_node):
                    conns[i].append(i)
            else:
                assert False, "No such internode connection strategy"
            internode_conn = conns

        num_senders = len(internode_conn)
        # Number of outgoing connections is restricted to be the same for all sender GPUs
        num_dsts = -1
        for (src, dsts) in internode_conn.items():
            if num_dsts == -1:
                num_dsts = len(dsts)
            else:
                assert num_dsts == len(dsts)
        total_outgoing_links = num_dsts * num_senders
        beta_split_factor = total_outgoing_links / nics_per_node
        node_topology.remote_beta = int(node_topology.remote_beta * beta_split_factor)
        node_topology.remote_invbw = int(node_topology.remote_alpha + node_topology.remote_beta)

        gpus_to_sender_rev_map = cs_json["internode_sketch"]["gpus_to_sender_rev_map"] if "gpus_to_sender_rev_map" in cs_json["internode_sketch"] else None
        enforce_ordering = cs_json["internode_sketch"]["enforce_ordering"] if "enforce_ordering" in cs_json["internode_sketch"] else False
        internode_sketch = InterNode_Relay(
            internode_conn,
            gpus_to_sender_rev_map,
            enforce_ordering,
        )
    else:
        internode_conn = None
        internode_sketch = None

    multinode_sketch = MultiNode(["round-robin"], [1], [copies])

    symmetry = Symmetry(cs_json["symmetry_offsets"])

    if reduce:
        scheduling_heuristic = 12
    elif len(switches):
        scheduling_heuristic = 10
    elif copies > 1:
        scheduling_heuristic = 14
    else:
        scheduling_heuristic = 5

    hyperparameters = HyperParameter(
        cs_json["hyperparameters"]["input_chunkup"],
        scheduling_heuristic
    )

    route_sketch = RouteSketch(
        intranode_sketch,
        internode_sketch,
        multinode_sketch,
        symmetry,
        hyperparameters
    )
    # return route_sketch

    topology = TACCLTopology(
        name=node_topology.name,
        copies=copies,
        ngpus_per_node=ngpus_per_node,
        node_links=node_topology.links,
        node_invbws=node_topology.invbws,
        remote_invbw=node_topology.remote_invbw,
        remote_alpha=node_topology.remote_alpha,
        remote_beta=node_topology.remote_beta,
        internode_conn=internode_conn,
        switches=switches
    )

    return topology, route_sketch
