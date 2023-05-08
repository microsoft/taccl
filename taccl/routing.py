# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from gurobipy import GRB, Model, quicksum
from taccl.algorithm import *
from taccl.shortest_path_sets import *
from taccl.topologies.route_sketch import *
from taccl.utils import *
from time import time

verbose = True

class TACCLRouting:
    def __init__(self, topology, route_sketch, collective):
        self.topology = topology
        self.route_sketch = route_sketch
        self.collective = collective
        self.chunkup = self.route_sketch.hyperparameters.chunkup

    def latency(self, src, dst, l):
        return self.topology.get_invbw(src,dst)

    def _encode(self, opt):
        # print("topology", self.topology.name)
        # print(self.topology.links)
        # print(self.topology.invbws)
        C = self.collective.num_chunks
        R = self.collective.num_nodes
        M = 10000000 # big-M for maximum self.time between sends
        ST = 5000000 # self.time for unstarted starts
        SND = 10000000 # self.time for unsent sends
        L = self.topology.L

        opt.Params.MIPFocus = 1
        opt.Params.Method = 2
        opt.Params.NumericFocus = 3
        opt.Params.Threads = 12
        opt.Params.MIPGap = 1e-9
        opt.Params.TimeLimit = 1800
        opt.Params.IntegralityFocus = 1
        opt.Params.IntFeasTol = 1e-9
        opt.Params.FeasibilityTol = 1e-9

        mu = 0.01

        self.send = opt.addVars(C, R, R, L, name="send", vtype=GRB.CONTINUOUS, lb=0.0)
        self.start = opt.addVars(C, R, name="start", vtype=GRB.CONTINUOUS, lb=0.0)
        self.time = opt.addVar(name="time", vtype=GRB.CONTINUOUS)
        opt.addLConstr(self.time <= ST-1)
        self.is_sent = opt.addVars(C,R,R,L, name="is_sent", vtype=GRB.BINARY)
        if self.route_sketch.intranode.strategy == "minmax":
            self.max_link_util = opt.addVar(name="max_link_util", vtype=GRB.CONTINUOUS, lb=0.0)
        if self.route_sketch.intranode.strategy == "maxmin":
            self.min_link_util = opt.addVar(name="min_link_util", vtype=GRB.CONTINUOUS, lb=0.0)

        num_local_nodes = self.topology.num_nodes() // self.topology.copies

        opt.ModelSense = GRB.MINIMIZE

        # Don't send chunks over connections which are not linked
        for src in range(R):
            for dst in range(R):
                if dst not in self.topology.destinations(src):
                    for c in self.collective.chunks():
                        for l in range(L):
                            opt.addLConstr(self.is_sent[c,src,dst,l] == 0)
                            opt.addLConstr(self.send[c,src,dst,l] == SND)
                else:
                    num_links = self.topology.link(src,dst)
                    for c in self.collective.chunks():
                        for l in range(num_links, L):
                            opt.addLConstr(self.is_sent[c,src,dst,l] == 0)
                            opt.addLConstr(self.send[c,src,dst,l] == SND)


        num_local_nodes = self.topology.num_nodes() // self.topology.copies
        for c in self.collective.chunks():
            for r in self.collective.ranks():
                opt.addLConstr(self.start[c,r] <= ST)
                # Fixing to only spsets will reduce chances for contiguity, but it is fine
                # Don't send to r if it is not in spset of c
                for src in self.topology.sources(r):
                    if (r not in self.spsets[c]) or (src not in self.spsets[c]):
                        for l in range(L):
                            opt.addLConstr(self.send[c,src,r,l] == SND)
                            opt.addLConstr(self.is_sent[c,src,r,l] == 0)
                if r not in self.spsets[c]:
                    opt.addLConstr(self.start[c,r] == ST)
                    continue

                if self.collective.precondition(r, c):
                    # Have chunks start on their starting ranks before the first step
                    opt.addLConstr(self.start[c,r] == 0)
                    for src in self.topology.sources(r):
                        for l in range(self.topology.link(src,r)):
                            opt.addLConstr(self.is_sent[c,src,r,l] == 0)
                else:
                    for src in self.topology.sources(r):
                        for l in range(self.topology.link(src,r)):
                            opt.addGenConstrIndicator(self.is_sent[c,src,r,l], True, self.start[c,r] == self.send[c,src,r,l] + self.latency(src,r,l))
                            opt.addGenConstrIndicator(self.is_sent[c,src,r,l], False, self.send[c,src,r,l] == SND)

                    if self.collective.postcondition(r, c):
                        opt.addLConstr(quicksum(quicksum(self.is_sent[c,src,r,l] for l in range(L)) for src in self.topology.sources(r)) == 1, name=f'post_{r}_{c}')
                        # opt.addLConstr(quicksum(quicksum(self.is_sent[c,src,r,l] for l in range(L)) for src in range(R)) == 1)
                        opt.addLConstr(self.start[c,r] <= self.time)
                    else:
                        opt.addLConstr(quicksum(quicksum(self.is_sent[c,src,r,l] for l in range(L)) for src in range(R)) <= 1, name=f'non_post_{r}_{c}')
                        opt.addLConstr(self.start[c,r] <= self.time + M*(1-quicksum(quicksum(self.is_sent[c,src,r,l] for l in range(L)) for src in range(R))))
                        opt.addLConstr(self.start[c,r] >= self.time + 1 - M*(quicksum(quicksum(self.is_sent[c,src,r,l] for l in range(L)) for src in range(R))))

                for src in self.topology.sources(r):
                    for l in range(self.topology.link(src,r)):
                        opt.addLConstr(self.start[c,src] <= self.send[c,src,r,l])

        # Count total switch send and switch recv in bounding the time of algo
        for l, switches in enumerate(self.topology.switches):
            for srcs, dsts, _, swtbw, switch_name in switches:
                if "in" in switch_name:
                    for dst in dsts:
                        opt.addLConstr(self.time >= quicksum(quicksum(swtbw*self.is_sent[c,srci,dst,l] for c in range(C)) for srci in srcs), name=f'switchin_{dst}_{l}')
                        if self.route_sketch.intranode.strategy == "minmax":
                            opt.addLConstr(self.max_link_util >= quicksum(quicksum(swtbw*self.is_sent[c,srci,dst,l] for c in range(C)) for srci in srcs), name=f'Mx_switchin_{dst}_{l}')
                        if self.route_sketch.intranode.strategy == "maxmin":
                            opt.addLConstr(self.min_link_util <= quicksum(quicksum(swtbw*self.is_sent[c,srci,dst,l] for c in range(C)) for srci in srcs), name=f'mx_switchin_{dst}_{l}')
                if "out" in switch_name:
                    for src in srcs:
                        opt.addLConstr(self.time >= quicksum(quicksum(swtbw*self.is_sent[c,src,dsti,l] for c in range(C)) for dsti in dsts), name=f'switchout_{src}_{l}')
                        if self.route_sketch.intranode.strategy == "minmax":
                            opt.addLConstr(self.max_link_util >= quicksum(quicksum(swtbw*self.is_sent[c,src,dsti,l] for c in range(C)) for dsti in dsts), name=f'Mx_switchout_{src}_{l}')
                        if self.route_sketch.intranode.strategy == "maxmin":
                            opt.addLConstr(self.min_link_util <= quicksum(quicksum(swtbw*self.is_sent[c,src,dsti,l] for c in range(C)) for dsti in dsts), name=f'mx_switchout_{src}_{l}')
                        for c in self.collective.chunks():
                            if src in self.spsets[c]:
                                for dstj in dsts:
                                    opt.addGenConstrIndicator(self.is_sent[c,src,dstj,l], True, self.time >= self.start[c,src] + swtbw * quicksum(self.is_sent[c,src,dsti,l] for dsti in dsts))
 
        # Count total link transfer in bounding the time of algo
        for r in self.collective.ranks():
            for src in self.topology.sources(r):
                for l in range(self.topology.link(src,r)):
                    opt.addLConstr(self.time >= quicksum(self.latency(src,r,l)*self.is_sent[c,src,r,l] for c in range(C)))
                    if self.route_sketch.intranode.strategy == "minmax":
                        opt.addLConstr(self.max_link_util >= quicksum(self.latency(src,r,l)*self.is_sent[c,src,r,l] for c in range(C)))
                    if self.route_sketch.intranode.strategy == "maxmin":
                        opt.addLConstr(self.min_link_util <= quicksum(self.latency(src,r,l)*self.is_sent[c,src,r,l] for c in range(C)))

        if isinstance(self.route_sketch.intranode, IntraNode_Switch):
            self._add_min_max_unique(opt, num_local_nodes, mu, L)

        if self.topology.copies > 1:
            self._add_relay_relaxation(opt, SND)
        if self.route_sketch.internode.enforce_ordering:
            self._enforce_ordering(opt)

        self._add_symmetry(opt, L)

        if isinstance(self.route_sketch.intranode, IntraNode_Switch):
            if self.route_sketch.intranode.switch_hyperedge_strategy[0] == "uc-min":
                print("--- minUniqueSends")
                opt.setObjective(self.time + self.mu * self.unique_links, GRB.MINIMIZE)
            elif self.route_sketch.intranode.switch_hyperedge_strategy[0] == "uc-max":
                print("--- maxUniqueSends")
                opt.setObjective(self.time - self.mu * self.unique_links, GRB.MINIMIZE)
            else:
                pass
        elif self.route_sketch.intranode.strategy == "minmax":
            print("minimizing maximum link utilization")
            opt.setObjective(self.time + self.mu * self.max_link_util, GRB.MINIMIZE)
        elif self.route_sketch.intranode.strategy == "maxmin":
            print("maximizing minimum link utilization") # To do better load balancing
            opt.setObjective(self.time - self.mu * self.min_link_util, GRB.MINIMIZE)
        else:
            opt.setObjective(self.time, GRB.MINIMIZE)

    def _enforce_ordering(self, opt):
        print("--- _enforce_ordering")
        assert self.route_sketch.internode.gpus_to_sender_rev_map is not None
        # Send the chunks of inter-node sender first and then the chunks of other gpus that are mapped to the inter-node sender
        sender_to_gpu = self.route_sketch.internode.gpus_to_sender_rev_map
        for sender in sender_to_gpu:
            for cp in range(self.topology.copies):
                src = self.topology.base_gpus[cp] + int(sender)
                all_chunks = [c for gpu in sender_to_gpu[sender] for c in self.collective.pre_chunk(self.topology.base_gpus[cp] + gpu)]
                sender_chunks = [c for c in self.collective.pre_chunk(src)]
                for r in self.topology.destinations(src):
                    if self.topology.gpu_to_node(r) != self.topology.gpu_to_node(src):
                        for c in all_chunks:
                            if c not in sender_chunks:
                                for c_sender in sender_chunks:
                                    for l in range(self.topology.link(src,r)):
                                        opt.addGenConstrIndicator(self.is_sent[c_sender,src,r,l], True, self.send[c,src,r,l] >= self.send[c_sender,src,r,l] + self.latency(src,r,l))

    def sym_rank(self, r, i, sym_offset, sym_size):
        return (r % sym_size + i * sym_offset) % sym_size + (r // sym_size) * sym_size

    def sym_chunk(self, c, i, sym_offset, sym_size):
        c_offset = c % self.chunkup
        # This method of find symmetric chunk works for Alltoall and Allgather
        # For Alltoall, there is a single pre and post rank for each chunk
        # For Allgather, there is a single pre rank for each chunk, thus still allowing a quick match
        for r_pre in self.collective.pre_rank(c):
            break
        for r_post in self.collective.post_rank(c):
            break
        r_pre_sym = self.sym_rank(r_pre, i, sym_offset, sym_size)
        r_post_sym = self.sym_rank(r_post, i, sym_offset, sym_size)
        c_sym = -1
        for c_opt in self.collective.post_chunk(r_post_sym):
            if self.collective.precondition(r_pre_sym, c_opt) and c_opt % self.chunkup == c_offset:
                assert c_sym == -1
                c_sym = c_opt
        return c_sym, r_pre_sym, r_post_sym

    def _add_symmetry(self, opt, L):
        print("--- _add_symmetry")
        num_nodes = self.topology.num_nodes()
        count = len(self.route_sketch.symmetry.offsets)
        for (sym_offset, sym_size) in self.route_sketch.symmetry.offsets:
            already_added = []
            for c in self.collective.chunks():
                c_sym, r_pre_sym, r_post_sym = self.sym_chunk(c, 1, sym_offset, sym_size)
                if c_sym == -1:
                    assert False, "Collective is not symmetric"
                pair_c = (c, c_sym) if c <= c_sym else (c_sym, c)
                if pair_c in already_added:
                    continue
                already_added.append(pair_c)
                for r in range(num_nodes):
                    r_sym = self.sym_rank(r, 1, sym_offset, sym_size)
                    for src in range(num_nodes):
                        if (r // sym_size == src // sym_size):
                            src_sym = self.sym_rank(src, 1, sym_offset, sym_size)
                            for l in range(L):
                                opt.addLConstr(self.send[c,src,r,l] == self.send[c_sym, src_sym, r_sym, l])
                                opt.addLConstr(self.is_sent[c,src,r,l] == self.is_sent[c_sym, src_sym, r_sym, l], name=f'sym_{c}_{src}_{r}_{src_sym}_{r_sym}_{l}')
                    opt.addLConstr(self.start[c,r] == self.start[c_sym, r_sym])


    def _add_relay_relaxation(self, opt, SND):
        print("--- _add_relay_relaxation_new")
        num_local_nodes = self.topology.num_nodes() // self.topology.copies
        chunk_to_sender_map = defaultdict(list)
        if self.route_sketch.internode.gpus_to_sender_rev_map is not None:
            for sender in self.route_sketch.internode.gpus_to_sender_rev_map:
                for gpu_src in self.route_sketch.internode.gpus_to_sender_rev_map[sender]:
                    for i in range(self.topology.copies):
                        node_sender = int(sender) + self.topology.base_gpus[i]
                        node_src = gpu_src + self.topology.base_gpus[i]
                        for c in self.collective.pre_chunk(node_src):
                            chunk_to_sender_map[c].append(node_sender)

        for (strategy, nnodes, group_size) in zip(self.route_sketch.multinode.strategy, self.route_sketch.multinode.nnodes, self.route_sketch.multinode.group_size):
            if strategy == "round-robin" or strategy == "relay":
                all_gpus = defaultdict()
                num_groups = self.topology.copies // group_size
                for base_n in range(0, group_size*num_groups, nnodes):
                    # print("base_n", base_n, "nnodes", nnodes, "group_size", group_size, "num_groups", num_groups)
                    all_gpus[base_n] = [g for g in range(self.topology.base_gpus[base_n], self.topology.base_gpus[base_n + nnodes])]
                for c in self.collective.chunks():
                    pair_set = defaultdict(set)
                    for r1 in self.collective.pre_rank(c):
                        for r2 in self.collective.post_rank(c):
                            n1 = self.topology.gpu_to_node(r1)
                            n2 = self.topology.gpu_to_node(r2)
                            base_n1 = (n1 // nnodes) * nnodes
                            base_n2 = (n2 // nnodes) * nnodes
                            if (base_n1 != base_n2) and (n1 // group_size == n2 // group_size):
                                senders = all_gpus[base_n1]
                                receivers = all_gpus[base_n2]
                                if self.route_sketch.internode.gpus_to_sender_rev_map is not None:
                                    assert c in chunk_to_sender_map
                                    assert len(set(chunk_to_sender_map[c]) & set(senders)) == len(set(chunk_to_sender_map[c]))
                                    senders = [g for g in chunk_to_sender_map[c]]
                                # remove senders and receivers that are not in spsets
                                senders = list(filter(lambda x: x in self.spsets[c], senders))
                                receivers = list(filter(lambda x: x in self.spsets[c], receivers))
                                for s in senders:
                                    for r in receivers:
                                        if self.topology.link(s,r) > 0:
                                            pair_set[(base_n1,base_n2)].add((s,r))
                    # print("pair set", c, pair_set)
                    for (bn1, bn2) in pair_set:
                        opt.addLConstr(quicksum(self.is_sent[c,src,r,l] for (src,r) in pair_set[(bn1,bn2)] for l in range(self.topology.link(src,r))) >= 1)
                        for src in all_gpus[bn1]:
                            for r in all_gpus[bn2]:
                                if (src,r) not in pair_set[(bn1,bn2)]:
                                    for l in range(self.topology.link(src,r)):
                                        opt.addLConstr(self.send[c,src,r,l] == SND)
                                        opt.addLConstr(self.is_sent[c,src,r,l] == 0, name=f'relay_notSend_{c}_{src}_{r}_{l}')
                                assert (r,src) not in pair_set[(bn1,bn2)]
                                for l in range(self.topology.link(r,src)):
                                    opt.addLConstr(self.send[c,r,src,l] == SND)
                                    opt.addLConstr(self.is_sent[c,r,src,l] == 0, name=f'relay_notSend_{c}_{r}_{src}_{l}')
                    # If c doesn't need to be sent outside the node, then set all internode transfers for that chunk to 0
                    if len(pair_set) == 0:
                        for r1 in self.collective.pre_rank(c):
                            n1 = self.topology.gpu_to_node(r1)
                            base_n1 = (n1 // nnodes) * nnodes
                            for src in self.collective.ranks():
                                for r in self.collective.ranks():
                                    if (src not in all_gpus[base_n1]) or (r not in all_gpus[base_n1]):
                                        for l in range(self.topology.link(src,r)):
                                            opt.addLConstr(self.send[c,src,r,l] == SND)
                                            opt.addLConstr(self.is_sent[c,src,r,l] == 0, name=f'relay_sendNotNeeded_{c}_{src}_{r}_{l}')

                            break

            elif strategy == "ring":
                assert False, "Ring strategy is not yet implemented"
            else:
                assert False, "strategy is not defined"

    def _add_min_max_unique(self, opt, num_local_nodes, mu, L):
        print("--- _add_min_max_unique")
        # print("SEND_AT_ALL")
        self.send_at_all = opt.addVars(num_local_nodes,num_local_nodes,L, name="send_at_all", vtype=GRB.BINARY)

        for r in range(num_local_nodes):
            for src in self.topology.sources(r):
                if src < num_local_nodes:
                    for l in range(L):
                        for c in self.collective.chunks():
                            opt.addLConstr(self.send_at_all[src,r,l] >= self.is_sent[c,src,r,l])
                        opt.addLConstr(self.send_at_all[src,r,l] <= quicksum(self.is_sent[c,src,r,l] for c in self.collective.chunks()))
        # print("mu", mu)
        self.mu = mu
        self.unique_links = quicksum(self.send_at_all[src,r,l] for l in range(L) for r in range(num_local_nodes) for src in self.topology.sources(r) if src < num_local_nodes)

    def optimize(self, distribute_over_links):
        import pickle as pkl
        heuristic = self.route_sketch.hyperparameters.heuristic
        # print("HEURISTIC:", heuristic)
        # print("finding shortest path sets")
        self.spsets = shortest_path_sets(self.topology, self.collective)

        # print(self.spsets)
        # print("found shortest path sets")
        instance_name = 'sccl_{}_{}_gurobiSimple'.format(self.topology.name, self.collective.name)

        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = self.topology.L

        start_time = time()
        opt = Model(instance_name)
        self._encode(opt)
        opt.optimize()
        end_time = time()
        print("simple time (encode+solve)", end_time-start_time, flush=True)

        opt.write(f'model_{instance_name}.lp')
        if opt.status == GRB.INFEASIBLE:
            opt.computeIIS()
            opt.write(f'model_{instance_name}.ilp')
            raise ValueError("Infeasible model")


        num_sols = 1
        for sol_i in range(num_sols):
            opt.Params.SolutionNumber = sol_i
            time_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
            chunk_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
            time_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
            chunk_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]

            model_str = ""
            for c in range(C):
                for r in range(R):
                    if self.start[c,r].Xn <= self.time.Xn + 0.005:
                        model_str += f'start[{c},{r}]={self.start[c,r].Xn}\n'
            dist_link_heuristic = [3,5,8,9,10,11,13] # Distribute chunk sends if there are multiple links connecting src to r
            if distribute_over_links:
                assert heuristic in dist_link_heuristic
            for c in range(C):
                sratch_str = defaultdict(list)
                for r in range(R):
                    for src in self.topology.sources(r):
                        for l in range(L):
                            if self.is_sent[c,src,r,l].Xn >= 0.995:
                                t_val = self.send[c,src,r,l].Xn
                                sratch_str[t_val].append(f'{c}: {src} --{l}--> {r}  t={self.send[c,src,r,l].Xn}\n')
                                # model_str += f'{c}: {src} --{l}--> {r}  t={self.send[c,src,r,l].Xn}\n'
                                if distribute_over_links:
                                    chunk_send[src][r][0].append(c)
                                    time_send[src][r][0].append(int(self.send[c,src,r,l].Xn + 0.005))
                                    chunk_recv[r][src][0].append(c)
                                    time_recv[r][src][0].append(int(self.start[c,r].Xn + 0.005))
                                else:
                                    chunk_send[src][r][l].append(c)
                                    time_send[src][r][l].append(int(self.send[c,src,r,l].Xn + 0.005))
                                    chunk_recv[r][src][l].append(c)
                                    time_recv[r][src][l].append(int(self.start[c,r].Xn + 0.005))
                for tval in sorted(sratch_str.keys()):
                    for strval in sratch_str[tval]:
                        model_str += strval
            # NOTE: we round the start and send times so integer here.
            # Would be good to have integral latencies for the path encoding
            # print(model_str)
            time_new = int(time())
            print(f"Saving cs_ts_cr_tr_simple_{time_new}")
            with open(f'cs_ts_cr_tr_simple_{time_new}.pkl', 'wb') as f:
                pkl.dump([chunk_send, time_send, chunk_recv, time_recv], f)

        return chunk_send, time_send, chunk_recv, time_recv

    def check_heuristic(self, topology, route_sketch, collective, ts_heur):
        import pickle as pkl
        assert ts_heur is not None
        print(f"Checking sol obtained by heuristic {route_sketch.hyperparameters.heuristic} ts={ts_heur}")
        with open(f'cs_ts_cr_tr_simple_{ts_heur}.pkl', 'rb') as f:
            chunk_send, time_send, chunk_recv, time_recv = pkl.load(f)
