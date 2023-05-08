# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from taccl import topologies
from taccl.algorithm import *
from taccl.heuristic_ordering import HeuristicOrderer
from taccl.instance import *
from taccl.shortest_path_sets import *
from gurobipy import GRB, Model, quicksum, abs_, and_
from taccl.utils import *
import numpy as np

class TACCLRevScheduler(object):
    def __init__(self, topology, route_sketch, collective):
        self.topology = topology
        self.route_sketch = route_sketch
        self.collective = collective
    
    def latency(self, src, dst, l):
        return self.topology.get_invbw(src,dst)

    def _is_relay_link(self,r,dst):
        if self.topology.gpu_to_node(r) != self.topology.gpu_to_node(dst):
            return True
        return False

    def _encode(self, opt, chunk_order, chunk_time,
        switch_chunk_order_recv, switch_chunk_time_recv, switch_chunk_order_send, switch_chunk_time_send,
        nic_chunk_order_recv, nic_chunk_time_recv, nic_chunk_order_send, nic_chunk_time_send,
        switch_link_mapping_recv=None, switch_link_mapping_send=None,
        endpoints_rc=[], prefer_local_reduce_first=True, extra_heuristic=True):

        # self.spsets = shortest_path_sets(self.topology, self.collective)
        heuristic = self.route_sketch.hyperparameters.heuristic
        self.chunkup = self.route_sketch.hyperparameters.chunkup
        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = self.topology.L
        smallM = 10
        M = 10000000 # big-M for maximum self.time between sends
        ST = 5000000 # self.time for unsent sends and unstarted starts
        SND = 10000000 # self.time for unsent sends and unstarted starts
        opt.Params.Threads = 4
        opt.Params.IntegralityFocus = 1

        self.is_sent_set_1 = set()
        self.is_before_set_1 = set()    # Fixed ordering of chunks over a link
        self.is_together_set_0 = set()  # Fix if chunks are received together on a GPU
        self.is_together_set_1 = set()
        self.recv_first_set_1 = set()   # Fixed ordering between recvs on a switch 
        self.nic_recv_first_set_1 = set()   # Fixed ordering between recvs on a NIC
        self.send_first_set_1 = set()   # Fixed ordering between sends on a switch
        self.nic_send_first_set_1 = set()   # Fixed ordering between sends on a NIC

        self.is_before = {}     # (c,o,r): c is received on GPU r before o, but from same source
        self.is_together = {}   # (c,o,r): c<o, c & o recvd together from same source on GPU r
        self.recv_first = {}
        self.send_first = {}
        self.is_reduce_before = {}

        self.send = opt.addVars(C, R, R, L, name="send", vtype=GRB.CONTINUOUS, lb=0.0)
        self.start = opt.addVars(C, R, name="start", vtype=GRB.CONTINUOUS, lb=0.0)
        self.time = opt.addVar(name="time", vtype=GRB.CONTINUOUS)

        # opt.setObjective(self.time, GRB.MINIMIZE)
        opt.addLConstr(self.time <= ST-1)

        num_local_nodes = R // self.topology.copies

        def minmax(c,o):
            if c <= o:
                return (c,o)
            else:
                return (o,c)

        # Try to send chunk c and o contiguously over r -> dst only if r -> dst is an IB
        def _should_try_together(r,dst,c,o):
            if (self.topology.copies <= 1):
                return False
            assert r != dst
            if self._is_relay_link(r,dst):
                # for allgather, not contig unless they are the input of the same GPU
                if "Allgather" in self.collective.name and (c//self.chunkup != o//self.chunkup):
                    return False
                return True
            return False

        # Can fix contiguous sends if reqd
        def _should_fix_together(r,dst,c,o):
            return False
            if not (isinstance(self.topology, DistributedTopology) and self.topology.m_top == MachineTopology.RELAYED):
                return False
            for rc in self.collective.pre_on(c):
                r1 = rc
            for ro in self.collective.pre_on(o):
                r2 = ro
            assert r != dst
            if self._is_relay_link(r,dst):
                if self.topology.bw_dist[rc][r] == self.topology.bw_dist[ro][r]:
                    return True
            return False

        # Populate is_sent_set_1 from the chunk_order received from path encoding
        def _add_chunk_sent(opt, heuristic):
            if chunk_order is not None:
                assert chunk_time is not None
                assert len(chunk_order) == R
                assert len(chunk_order[0]) == R
                assert len(chunk_order[0][0]) <= L
                # TODO: we will do distribution at heuristic_ordering.py
                # dist_link_heuristic = [3,5,8,9,10,11,13,12] # Distribute chunk sends if there are multiple links connecting src to r
                for r in range(R):
                    for src in self.topology.sources(r):
                        for l in range(self.topology.link(src,r)):
                            for c in chunk_order[r][src][l]:
                                self.is_sent_set_1.add((c,src,r,l))

        def _add_switch_order(switch_chunk_order_recv, switch_chunk_order_send, switch_link_mapping_recv, switch_link_mapping_send):
            # Order recvs coming into and going out from a GPU connected to a switch
            # recv_right_after[r][(c,srci)] = (o,srcj) => GPU r receives o from srcj right after receiving c from srci
            # send_right_after[r][(c,dsti)] = (o,dstj) => GPU r sends o to dstj right after sending c to dsti
            # (c,o,r,srci,srcj) \in recv_first_set_1 => c is recvd on r from srci anytime before o is recvd on r from srcj
            # (c,o,ri,dsti,dstj) \in send_first_set_1 => c is sent from r to dsti anytime before o is sent from r to dstj
            LL = 0
            for r in range(R):
                LL = max(LL, len(switch_chunk_order_recv[r]))
                LL = max(LL, len(switch_chunk_order_send[r]))

            recv_right_after, recv_first_set_1, send_right_after, send_first_set_1 = add_switch_order(switch_chunk_order_recv,
                switch_chunk_order_send,
                switch_link_mapping_recv,
                switch_link_mapping_send, R, LL)
            for recv in recv_first_set_1:
                self.recv_first_set_1.add(recv)
            for send in send_first_set_1:
                self.send_first_set_1.add(send)
            return recv_right_after, send_right_after

        def _add_chunk_order(opt, heuristic, recv_right_after, send_right_after):
            # dist_link_heuristic = [3,5,8,9,10,12, 13] # Distribute chunk sends if there are multiple links connecting src to r
            if chunk_order is not None:
                assert chunk_time is not None
                assert len(chunk_order) == R
                assert len(chunk_order[0]) == R
                for r in range(R):
                    for src in self.topology.sources(r):
                        if len(chunk_order[r][src][0]):
                            print("chunk_order", r, src, chunk_order[r][src][0])
                        for l in range(self.topology.link(src,r)):
                            this_chunk_order = chunk_order[r][src][l]
                            max_contig = 6
                            for i, c in enumerate(this_chunk_order):
                                j = i + 1
                                while j<len(this_chunk_order):
                                    o = this_chunk_order[j]
                                    c1,o1 = minmax(c,o)
                                    prev_o = this_chunk_order[j-1]
                                    c2, prev_o2 = minmax(c,prev_o)
                                    skip_others = False
                                    if self._is_relay_link(src,r) and (src,r) in self.topology.switches_involved:
                                        ll = self.topology.switch_map_rev[r][(src,l)]
                                        ll_src = self.topology.switch_map_send[src][(r,l)]
                                        assert len(recv_right_after[r][ll]) > 0
                                        if (recv_right_after[r][ll][(c,src)] != recv_right_after[r][ll][(o,src)] or send_right_after[src][ll_src][(c,r)] != send_right_after[src][ll_src][(o,r)]):
                                            self.is_together_set_0.add((c1,o1,r,src))
                                            self.is_before_set_1.add((c,o,r,src))
                                            skip_others = True
                                    if not skip_others:
                                        if _should_fix_together(src,r,c,o):
                                            self.is_together_set_1.add((c1,o1,r,src))
                                        # Max contiguity allowed = 6
                                        elif _should_try_together(src,r,c,o) and j-i<max_contig:
                                            # print(f'will-try {c} {o} ({src}->{r})')
                                            is_before_ocr = 0
                                            if not extra_heuristic:
                                                if (o,c,r,src) not in self.is_before:
                                                    self.is_before[(o,c,r,src)] = opt.addVar(vtype=GRB.BINARY)
                                                is_before_ocr = self.is_before[(o,c,r,src)]
                                            else:
                                                assert (o,c,r,src) not in self.is_before
                                            if (c,o,r,src) not in self.is_before:
                                                self.is_before[(c,o,r,src)] = opt.addVar(vtype=GRB.BINARY)
                                            if (c1,o1,r) not in self.is_together:
                                                self.is_together[(c1,o1,r,src)] = opt.addVar(vtype=GRB.BINARY)
                                            opt.addLConstr(self.is_before[(c,o,r,src)] + self.is_together[(c1,o1,r,src)] + is_before_ocr == 1)
                                            # send chunk together with another only if the previous chunk between the two has been sent together
                                            if j-1>i:
                                                opt.addLConstr(self.is_together[(c1,o1,r,src)] <= self.is_together[(c2,prev_o2,r,src)])
                                        else:
                                            self.is_together_set_0.add((c1,o1,r,src))
                                            self.is_before_set_1.add((c,o,r,src))
                                            # if c1 == 34 and o1 == 35:
                                            #     print("2. added is_before ", c,o,r,src)
                                    j = j + 1
                                i = i + 1

        def alpha(r,dst):
            assert r != dst
            if self._is_relay_link(r,dst):
                alpha = self.topology.remote_alpha
                assert alpha is not None
                return alpha
            return 0

        def beta(r,dst):
            assert r != dst
            if self._is_relay_link(r,dst):
                beta = self.topology.remote_beta
                assert beta is not None
                return beta
            return self.topology.get_invbw(r,dst)

        def calc_latency(src,r,l,c):
            if self._is_relay_link(src,r):
                num_s = 0
                for o in range(C):
                    o1,c1 = minmax(o,c)
                    if (o1,c1,r,src) in self.is_together_set_1:
                        assert (o1,c1,r,src) not in self.is_together
                        num_s = num_s + 1
                        continue
                    if (o1,c1,r,src) in self.is_together_set_0:
                        assert (o1,c1,r,src) not in self.is_together
                    else:
                        if (o1,c1,r,src) not in self.is_together:
                            self.is_together[(o1,c1,r,src)] = opt.addVar(vtype=GRB.BINARY)
                lat = alpha(src,r) + beta(src,r)*(num_s + quicksum(self.is_together[(o,c,r,src)] if (o,c,r,src) in self.is_together else 0 for o in range(c)) + quicksum(self.is_together[(c,o,r,src)] if (c,o,r,src) in self.is_together else 0 for o in range(c,C)))
                return lat
            return alpha(src,r) + beta(src,r)

        # Set chunk is_send_set
        _add_chunk_sent(opt, heuristic)
        
        # Populate values
        for c in self.collective.chunks():
            for r in self.collective.ranks():
                recvd_anytime = sum([sum([1 if (c,src,r,l) in self.is_sent_set_1 else 0 for l in range(L)]) for src in self.topology.sources(r)])
                recv_IB = sum([sum([1 if (c,src,r,l) in self.is_sent_set_1 and self._is_relay_link(src,r) else 0 for l in range(L)]) for src in self.topology.sources(r)])
                if recvd_anytime == 0:
                    for srci in self.topology.sources(r):
                        assert (c,c,r,srci) not in self.is_together_set_1
                        assert (c,c,r,srci) not in self.is_together
                        self.is_together_set_0.add((c,c,r,srci))
                else:
                    # Will receive a chunk at most once
                    # assert recvd_anytime == 1
                    for srci in self.topology.sources(r):
                        assert (c,c,r,srci) not in self.is_together_set_1
                        assert (c,c,r,srci) not in self.is_together
                        self.is_together_set_1.add((c,c,r,srci))

        # Set ordering
        should_add_switch_order = True
        recv_right_after = {}
        send_right_after = {}
        if should_add_switch_order:
            recv_right_after, send_right_after = _add_switch_order(
                                                    switch_chunk_order_recv,
                                                    switch_chunk_order_send,
                                                    switch_link_mapping_recv,
                                                    switch_link_mapping_send)

        _add_chunk_order(opt, heuristic, recv_right_after, send_right_after)

        # returns (is_static_val_cor, is_before_cor)
        def _get_isbefore(c,o,r,src):
            if (c,o,r,src) in self.is_before_set_1:
                return True, 1
            elif (c,o,r,src) in self.is_before:
                return False, self.is_before[(c,o,r,src)]
            else:
                return True, 0

        # returns (is_static_val_cor, is_together_cor)
        def _get_istogether(c,o,r,src):
            c1,o1 = minmax(c,o)
            if (c1,o1,r,src) in self.is_together_set_1:
                return True, 1
            elif (c1,o1,r,src) in self.is_together:
                return False, self.is_together[(c1,o1,r,src)]
            else:
                return True, 0

        print("endpoints_rc", endpoints_rc)
        # Correctness constraints
        self.weighted_terms_to_min = []
        for r in self.collective.ranks():
            src_r = [src for src in self.topology.sources(r)]
            links_r = {src: self.topology.link(src,r) for src in src_r}
            for c in self.collective.chunks():
                opt.addLConstr(self.start[c,r] <= ST)
                if (r,c) in endpoints_rc:
                    opt.addLConstr(self.start[c,r] == 0)
                else:
                    # Bandwidth constraint
                    for src in src_r:
                        for l in range(links_r[src]):
                            if (c,src,r,l) in self.is_sent_set_1:
                                opt.addLConstr(self.start[c,r] >= self.send[c,src,r,l] + calc_latency(src,r,l,c))
                            else:
                                opt.addLConstr(self.send[c,src,r,l] >= SND)
                        for l in range(links_r[src], L):
                            opt.addLConstr(self.send[c,src,r,l] == SND)
                    recvd_anytime = sum([sum([1 if (c,src,r,l) in self.is_sent_set_1 else 0 for l in range(links_r[src])]) for src in src_r])
                    if self.collective.precondition(r, c):
                        opt.addLConstr(self.start[c,r] <= self.time)
                    else:
                        if recvd_anytime == 0 and (r,c) not in endpoints_rc:
                            print("setting >", c,r)
                            opt.addLConstr(self.start[c,r] >= self.time + 1)
                        else:
                            opt.addLConstr(self.start[c,r] <= self.time)

                c_sources = []
                for src in src_r:
                    for l in range(links_r[src]):
                        if (c,src,r,l) in self.is_sent_set_1:
                            opt.addLConstr(self.start[c,src] <= self.start[c,r])
                            # c_sources.append((src,l))
                            c_sources.append(src) # NOTE assuming l == 0 always
                        opt.addLConstr(self.start[c,src] <= self.send[c,src,r,l])
                
                for i in range(len(c_sources)):
                    for j in range(len(c_sources)):
                        if i!=j:
                            srci = c_sources[i]
                            srcj = c_sources[j]
                            srci1, srcj1 = minmax(srci,srcj)
                            if (r,c,srci1,srcj1) not in self.is_reduce_before:
                                self.is_reduce_before[(r,c,srci1,srcj1)] = opt.addVar(vtype=GRB.BINARY)
                                if (r//num_local_nodes == srci1//num_local_nodes) and (r//num_local_nodes != srcj1//num_local_nodes):
                                    # try to reduce local nodes first (but only try)
                                    self.weighted_terms_to_min.append(-self.is_reduce_before[(r,c,srci1,srcj1)])
                                elif (r//num_local_nodes == srcj1//num_local_nodes) and (r//num_local_nodes != srci1//num_local_nodes):
                                    self.weighted_terms_to_min.append(self.is_reduce_before[(r,c,srci1,srcj1)])
                                
                                opt.addGenConstrIndicator(self.is_reduce_before[(r,c,srci1,srcj1)], True, self.send[c,srcj1,r,0] >= self.send[c,srci1,r,0] + calc_latency(srci1,r,l,c))
                                opt.addGenConstrIndicator(self.is_reduce_before[(r,c,srci1,srcj1)], False, self.send[c,srci1,r,0] >= self.send[c,srcj1,r,0] + calc_latency(srcj1,r,l,c))


                # Order sends from same gpu to same gpu
                for o in range(c):
                    for src in src_r:
                        is_static_cor, is_before_cor = _get_isbefore(c,o,r,src)
                        is_static_ocr, is_before_ocr = _get_isbefore(o,c,r,src)
                        is_static_t_ocr, is_together_ocr = _get_istogether(o,c,r,src)
                        # chunks sent together must have same send and start time
                        if is_static_t_ocr and is_together_ocr == 1:
                            for l in range(self.topology.link(src,r)):
                                if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1:
                                    opt.addLConstr(self.send[c,src,r,l] == self.send[o,src,r,l])
                            opt.addLConstr(self.start[c,r] == self.start[o,r])
                        elif not is_static_t_ocr:
                            for l in range(self.topology.link(src,r)):
                                if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1:
                                    opt.addGenConstrIndicator(self.is_together[(o,c,r,src)], True, self.send[c,src,r,l] == self.send[o,src,r,l])


                        if is_static_cor and is_static_ocr and is_static_t_ocr:
                            sent_same = any([1 if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1 else 0 for l in range(L)])
                            sent_val = 1 if sent_same else 0
                            assert is_before_cor + is_before_ocr + is_together_ocr == sent_val, f'{c}, {o}, {r}, {is_before_cor}, {is_before_ocr}, {is_together_ocr}, {sent_val}'

                    # Bandwidth constraints based on chunk send times
                        for l in range(self.topology.link(src,r)):
                            if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1:
                                lat_o = calc_latency(src,r,l,o)
                                lat_c = calc_latency(src,r,l,c)

                                if (c,o,r,src) in self.is_before_set_1:
                                    # print(c,"is_before",o, "for", src, "to", r)
                                    opt.addLConstr(self.send[c,src,r,l] + lat_c <= self.send[o,src,r,l])
                                elif (c,o,r,src) in self.is_before:
                                    # print(c,"may be before",o, "for", src, "to", r)
                                    opt.addLConstr(self.send[c,src,r,l] + lat_c <= self.send[o,src,r,l] + M*(1-self.is_before[(c,o,r,src)]))
                                if (o,c,r,src) in self.is_before_set_1:
                                    # print(o,"is_before",c, "for", src, "to", r)
                                    opt.addLConstr(self.send[o,src,r,l] + lat_o <= self.send[c,src,r,l])
                                elif (o,c,r,src) in self.is_before:
                                    # print(o,"may be before",c, "for", src, "to", r)
                                    opt.addLConstr(self.send[o,src,r,l] + lat_o <= self.send[c,src,r,l] + M*(1-self.is_before[(o,c,r,src)]))

        # Order receives from a switch
        for (c,src,r,l) in self.is_sent_set_1:
            if (src,r) in self.topology.switches_involved:
                for swt_i, swt_type in self.topology.switches_involved[(src,r)]:
                    srcs_check = []
                    if l == swt_i:
                        for srcs, dsts, _, _, switch_name in self.topology.switches[swt_i]:
                            if r in dsts and "in" in switch_name and src in srcs:
                                srcs_check = srcs
                                assert len(srcs_check)>0, f'{r} {c} {src} {l} {self.topology.switches[l]}'
                                break
                        lat_c = calc_latency(src,r,l,c)
                        for o in range(c):
                            for src_o in srcs_check:
                                if src_o == src:
                                    continue
                                if (o,src_o,r,l) in self.is_sent_set_1:
                                    if o == c:
                                        assert False
                                    lat_o = calc_latency(src_o,r,l,o)
                                    if (o,c,r,l,src_o,src) in self.recv_first_set_1:
                                        opt.addLConstr(self.send[o,src_o,r,l] + lat_o <= self.send[c,src,r,l])
                                    elif (c,o,r,l,src,src_o) in self.recv_first_set_1:
                                        opt.addLConstr(self.send[c,src,r,l] + lat_c <= self.send[o,src_o,r,l])
                                    else:
                                        assert False, f"no-ordering {o}, {c}, {r}, {src}, {src_o}"
                                        assert (o,c,r,l) not in self.recv_first, f'{o},{c},{r},{l}'
                                        self.recv_first[(o,c,r,l)] = opt.addVar(vtype=GRB.BINARY)
                                        opt.addLConstr(self.start[o,r] + lat_c <= self.start[c,r] + M*(1-self.recv_first[(o,c,r,l)]))
                                        opt.addLConstr(self.start[c,r] + lat_o <= self.start[o,r] + M*(self.recv_first[(o,c,r,l)]))

        # Order sends to a switch
        for (c,r,dst,l) in self.is_sent_set_1:
            if (r,dst) in self.topology.switches_involved:
                for swt_i, swt_type in self.topology.switches_involved[(r,dst)]:
                    dsts_check = []
                    if l == swt_i:
                        for srcs, dsts, _, _, switch_name in self.topology.switches[swt_i]:
                            if r in srcs and "out" in switch_name and dst in dsts:
                                dsts_check = dsts
                                assert len(dsts_check)>0, f'{r} {c} {dst} {l} {self.topology.switches[l]}'
                                break
                        lat_c = calc_latency(r,dst,l,c)
                        for o in range(c+1):
                            for dst_o in dsts_check:
                                if dst_o == dst:
                                    continue
                                if (o,r,dst_o,l) in self.is_sent_set_1:
                                    lat_o = calc_latency(r,dst_o,l,o)
                                    if (o,c,r,l,dst_o,dst) in self.send_first_set_1:
                                        opt.addLConstr(self.send[o,r,dst_o,l] + lat_o <= self.send[c,r,dst,l])
                                    elif (c,o,r,l,dst,dst_o) in self.send_first_set_1:
                                        opt.addLConstr(self.send[c,r,dst,l] + lat_c <= self.send[o,r,dst_o,l])
                                    else:
                                        assert False
                                        assert (o,c,r,l) not in self.send_first, f'{o},{c},{r},{l}'
                                        self.send_first[(o,c,r,l)] = opt.addVar(vtype=GRB.BINARY)
                                        opt.addLConstr(self.send[o,r,dst_o,l] + lat_o <= self.send[c,r,dst,l] + M*(1-self.send_first[(o,c,r,l)]))
                                        opt.addLConstr(self.send[c,r,dst,l] + lat_c <= self.send[o,r,dst_o,l] + M*(self.send_first[(o,c,r,l)]))

        if prefer_local_reduce_first and len(self.weighted_terms_to_min):
            print("Weighted terms will be minimized:")
            print(self.weighted_terms_to_min)
            opt.setObjective(self.time + 0.001 * quicksum([term for term in self.weighted_terms_to_min]), GRB.MINIMIZE)
        else:
            opt.setObjective(self.time, GRB.MINIMIZE)

    def optimize_reversed(self, chunk_order=None, time_recv=None,
                switch_chunk_recv=None, switch_time_recv=None, switch_chunk_send=None, switch_time_send=None,
                nic_chunk_recv=None, nic_time_recv=None, nic_chunk_send=None, nic_time_send=None,
                switch_link_mapping_recv=None, switch_link_mapping_send=None, paths=None, prefer_local_reduce_first=False):
        
        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = self.topology.L
        
        from time import time
        endpoints_rc = []
        for c in paths:
            for path in paths[c]:
                last_transfer_r = path[0][1]
                endpoints_rc.append((last_transfer_r,c))
        self.topology.reverse_links()

        start_time = time()
        opt = Model('taccl_{}_{}'.format(self.topology.name, self.collective.name))

        # call to _encode swaps the order of switch_link_mapping_send and switch_link_mapping_recv
        self._encode(opt, chunk_order, time_recv, 
            switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send,
            nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send,
            switch_link_mapping_send, switch_link_mapping_recv, endpoints_rc, self.route_sketch.hyperparameters.heuristic, prefer_local_reduce_first)
        opt.optimize()
        end_time = time()
        print("strict time (encode+solve)", end_time-start_time, flush=True)

        if opt.status == GRB.INFEASIBLE:
            opt.computeIIS()
            opt.write("model.ilp")
            raise ValueError("Infeasible model")

        send_dict = defaultdict(list)
        SCALE_TIME = 10

        model_str = ""
        other_model_str = ""
        for c in range(C):
            for r in range(R):
                if self.start[c,r].X <= self.time.X + 0.005:
                    model_str += f'start[{c},{r}]={self.start[c,r].X}\n'
        recv_times = defaultdict(list)
        chunk_path = [defaultdict(list) for c in range(C)]
        for src in range(R):
            for r in self.topology.destinations(src):
                for l in range(L):
                    for c_np in chunk_order[r][src][l]:
                        c = int(c_np)
                        assert (c,src,r,l) in self.is_sent_set_1
                        # model_str += f'{c}: {src} --{l}--> {r}  t={self.send[c,src,r,l].X}\n'
                        t = int(SCALE_TIME*self.send[c,src,r,l].X + 0.0001)
                        transfer_str = f'{c}: {src} --{l}--> {r}  t={self.send[c,src,r,l].X}\n'
                        recv_times[t].append(transfer_str)
                        chunk_path[c][t].append(transfer_str)
                        send_dict[t].append([c,src,r,t,l,'rrc'])
                    for c_np in range(C):
                        c = int(c_np)
                        if c not in chunk_order[r][src][l]:
                            assert (c,src,r,l) not in self.is_sent_set_1
        for tval in sorted(recv_times.keys()):
            for strval in recv_times[tval]:
                model_str += strval
        for c in range(C):
            for tval in sorted(chunk_path[c].keys()):
                for strval in chunk_path[c][tval]:
                    other_model_str += strval
        for c in range(C):
            for o in range(c):
                for r in range(R):
                    for src in self.topology.sources(r):
                        if (o,c,r,src) in self.is_together:
                            if self.is_together[(o,c,r,src)].X >= 0.995:
                                print(f'is_together[{o},{c},{r},{src}] = {self.is_together[(o,c,r,src)].X}')
                                model_str += f'({c},{o},{r},{src})\n'
                        elif (o,c,r,src) in self.is_together_set_1:
                            model_str += f'({c},{o},{r},{src}) set\n'
                            print(f'({c},{o},{r},{src}) set together')
                        if (c,o,r,src) in self.is_before and self.is_before[(c,o,r,src)].X >= 0.995:
                            print(f'is_before[{c},{o},{r},{src}]')
                        if (o,c,r,src) in self.is_before and self.is_before[(o,c,r,src)].X >= 0.995:
                            print(f'is_before[{o},{c},{r},{src}]')

        print(model_str)
        print("Chunk path:")
        print(other_model_str)
        return send_dict


    def build_allreduce(self, reduce_coll, send_dict_redscat, send_dict_allgather, ts):
        import math

        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = self.topology.L
        print(R,C,L)

        SCALE_TIME = 10

        do_redscat = True
        do_allgather = True

        # assert len(ts)
        # send_dict_allgather = np.load(f"send_dict_{ts}.npy", allow_pickle=True).item()
        steps=[]
        send_times_redscat = sorted(send_dict_redscat.keys())
        print("senddicts:")
        print("send_dict_redscat:", send_dict_redscat)
        print("send_dict_allgather:", send_dict_allgather)
        tmax = send_times_redscat[-1]
        print("tmax", tmax)
        shifted_send_dict_allgather = defaultdict(list)
        for t in send_dict_allgather:
            for (c,src,r,t_,l) in send_dict_allgather[t]:
                t_shifted = tmax + t + SCALE_TIME
                shifted_send_dict_allgather[t_shifted].append([c,src,r,t_shifted,l,None]) # reverse

        if do_redscat and do_allgather:
            send_dict = send_dict_redscat.copy()
            send_dict.update(shifted_send_dict_allgather)
        elif do_redscat:
            send_dict = send_dict_redscat.copy()
        elif do_allgather:
            send_dict = shifted_send_dict_allgather.copy()
            self.topology.reverse_links()
        print("send_dict:", send_dict)
        send_times = sorted(send_dict.keys())

        i = 0
        while(i < len(send_times)):
            num_sends = [[0 for _ in range(R)] for _ in range(R)]
            j = i + 1
            while j < len(send_times):
                to_break = False
                t_end = send_times[j]
                for (c,src,r,_,_,redop) in send_dict[t_end]:
                    for t in range(i,j):
                        for (ci,srci,ri,_,_,redopi) in send_dict[send_times[t]]:
                            if (c == ci and src == ri) or (c == ci and r == ri and redop is not None and redopi is not None):
                                to_break = True
                                break
                        if to_break:
                            break
                    if to_break:
                        break
                if to_break:
                    break
                j = j + 1
            sends = []
            for k in range(i,j):
                sends.extend(send_dict[send_times[k]])
            print(sends)
            num_sends = [[[0 for _ in range(L)] for _ in range(R)] for _ in range(R)]
            for (c,src,r,_,l,_) in sends:
                num_sends[r][src][l] = num_sends[r][src][l] + 1
            rounds = 0
            for srcs, dsts, bw, l, name in self.topology.real_bandwidth_constraints():
                util = 0
                for dst in dsts:
                    for src in srcs:
                        util += num_sends[dst][src][l]
                if rounds <= util * bw * SCALE_TIME:
                    rounds = math.ceil(util * bw * SCALE_TIME)
            step = Step(rounds, sorted(sends, key=lambda x: x[3]))
            print("STEP ", step)
            steps.append(step)
            i = j

        if do_allgather and do_redscat:
            instance = Instance(
                steps=len(steps),
                extra_rounds=0,
                chunks=R*self.chunkup,
            )
        elif do_redscat:
            instance = Instance(
                steps=len(steps),
                extra_rounds=0,
                chunks=self.chunkup,
            )
            for step in steps:
                print(step)
        elif do_allgather:
            instance = Instance(
                steps=len(steps),
                extra_rounds=0,
                chunks=self.chunkup,
            )

        if do_redscat and do_allgather:
            soltype = f"{ts}-allreduce"
        elif do_redscat:
            soltype = f"{ts}-redscat"
        elif do_allgather:
            soltype = f"{ts}-allgather"

        from time import time
        timestamp = int(time())
        np.save(f'send_dict_allred_{timestamp}.npy', send_dict)
        return Algorithm.make_implementation(reduce_coll, self.topology, instance, steps, cont=True, suffix=f'-gurobisol-{soltype}-{timestamp}')
