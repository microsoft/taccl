# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from taccl.algorithm import *
from taccl.instance import *
from taccl.shortest_path_sets import *
from gurobipy import GRB, Model, quicksum, abs_, and_
from taccl.utils import *
import numpy as np

class TACCLScheduler(object):
    def __init__(self, topology, route_sketch, collective):
        self.topology = topology
        self.route_sketch = route_sketch
        self.collective = collective

    # Don't care about relay relaxation - gurobi simple fixes that
    def _is_relay_link(self,r,dst):
        if self.topology.gpu_to_node(r) != self.topology.gpu_to_node(dst):
            return True
        return False

    def _encode(self, opt, chunk_order, chunk_time,
        switch_chunk_order_recv, switch_chunk_time_recv, switch_chunk_order_send, switch_chunk_time_send,
        nic_chunk_order_recv, nic_chunk_time_recv, nic_chunk_order_send, nic_chunk_time_send,
        switch_link_mapping_recv=None, switch_link_mapping_send=None, extra_heuristic=True):

        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = self.topology.L
        heuristic = self.route_sketch.hyperparameters.heuristic
        smallM = 10
        M = 10000000 # big-M for maximum self.time between sends
        ST = 500000 # self.time for unsent sends and unstarted starts
        SND = 1000000 # self.time for unsent sends and unstarted starts
        opt.Params.Threads = 1
        opt.Params.IntegralityFocus = 1
        opt.Params.IntFeasTol = 1e-9
        opt.Params.FeasibilityTol = 1e-9
        opt.Params.TimeLimit = 1800

        self.spsets = shortest_path_sets(self.topology, self.collective)
        num_local_nodes = R // self.topology.copies

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

        self.send = opt.addVars(C, R, R, L, name="send", vtype=GRB.CONTINUOUS, lb=0.0)
        self.start = opt.addVars(C, R, name="start", vtype=GRB.CONTINUOUS, lb=0.0)
        self.time = opt.addVar(name="time", vtype=GRB.CONTINUOUS)

        opt.setObjective(self.time, GRB.MINIMIZE)
        opt.addLConstr(self.time <= ST-1)

        def minmax(c,o):
            if c <= o:
                return (c,o)
            else:
                return (o,c)

        # Try to send chunk c and o contiguously over r -> dst only if r -> dst is an IB
        def _should_try_together(r,dst,c,o):
            assert r != dst
            if self._is_relay_link(r,dst):
                return True
            return False

        # Can fix contiguous sends if reqd
        def _should_fix_together(r,dst,l,c,o):
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
                for r in range(R):
                    for src in self.topology.sources(r):
                        for l in range(self.topology.link(src,r)):
                            for c in chunk_order[r][src][l]:
                                assert r in self.spsets[c]
                                self.is_sent_set_1.add((c,src,r,l))

        def _add_switch_order(switch_chunk_order_recv, switch_chunk_order_send, switch_link_mapping_recv, switch_link_mapping_send):
            # Creating new datastructures from order information
            # Order recvs coming into and going out from a GPU connected to a switch
            # Input:
            #   switch_chunk_order_recv[r][ll] : [(c1,src1), ...] in order
            #   switch_chunk_order_send[r][ll] : [(c1,dst1), ...] in order
            #   switch_link_mapping[r][ll] = l
            # Output:
            #   recv_right_after[r][ll][(c,srci)] = (o,srcj) => GPU r over switch l receives o from srcj right after receiving c from srci
            #   send_right_after[r][ll][(c,dsti)] = (o,dstj) => GPU r over switch l sends o to dstj right after sending c to dsti
            #       recv_ and send_ right_after give the first chunk received from / sent to a different GPU
            #   (c,o,r,l,srci,srcj) \in recv_first_set_1 => c is recvd on r from srci anytime before o is recvd on r from srcj
            #   (c,o,r,l,dsti,dstj) \in send_first_set_1 => c is sent from r to dsti anytime before o is sent from r to dstj
            #  Note that the l and ll are different for right_after and first_set
            LL = 0
            for r in range(R):
                LL = max(LL, len(switch_chunk_order_recv[r]))
                LL = max(LL, len(switch_chunk_order_send[r]))

            recv_right_after, recv_first_set_1, send_right_after, send_first_set_1 = add_switch_order(switch_chunk_order_recv,
                                                                                        switch_chunk_order_send, switch_link_mapping_recv, switch_link_mapping_send,
                                                                                        R, LL)
            for recv in recv_first_set_1:
                self.recv_first_set_1.add(recv)
            for send in send_first_set_1:
                self.send_first_set_1.add(send)
            return recv_right_after, send_right_after

        def _add_chunk_order(opt, heuristic, recv_right_after, send_right_after):
            if chunk_order is not None:
                assert chunk_time is not None
                assert len(chunk_order) == R
                assert len(chunk_order[0]) == R
                for r in range(R):
                    for src in self.topology.sources(r):
                        for l in range(self.topology.link(src,r)):
                            this_chunk_order = chunk_order[r][src][l]
                            max_contig = 6
                            for i, c in enumerate(this_chunk_order):
                                j = i + 1
                                is_input_i = self.collective.precondition(src,c)
                                while j<len(this_chunk_order):
                                    o = this_chunk_order[j]
                                    c1,o1 = minmax(c,o)
                                    prev_o = this_chunk_order[j-1]
                                    c2, prev_o2 = minmax(c,prev_o)
                                    # do not send input and scratch buffers together because they are not contiguous in memory anyway
                                    has_i_s_break = False
                                    for ii in range(i+1,j+1):
                                        o_ii = this_chunk_order[ii]
                                        if self.collective.precondition(src,o_ii) != is_input_i:
                                            has_i_s_break = True
                                            break
                                    assert r in self.spsets[c] and r in self.spsets[o]
                                    skip_others = False
                                    if self._is_relay_link(src,r) and (src,r) in self.topology.switches_involved:
                                        ll = self.topology.switch_map_rev[r][(src,l)]
                                        ll_src = self.topology.switch_map_send[src][(r,l)]
                                        assert len(recv_right_after[r][ll]) > 0
                                        if (recv_right_after[r][ll][(c,src)] != recv_right_after[r][ll][(o,src)] or send_right_after[src][ll_src][(c,r)] != send_right_after[src][ll_src][(o,r)]):
                                            self.is_together_set_0.add((c1,o1,r))
                                            self.is_before_set_1.add((c,o,r))
                                            skip_others = True
                                    if not skip_others:
                                        if heuristic == 11 and has_i_s_break:
                                            self.is_together_set_0.add((c1,o1,r))
                                            self.is_before_set_1.add((c,o,r))
                                        elif _should_fix_together(src,r,l,c,o):
                                            self.is_together_set_1.add((c1,o1,r))
                                        # Max contiguity allowed = 6
                                        elif _should_try_together(src,r,c,o) and j-i<max_contig:
                                            is_before_ocr = 0
                                            if not extra_heuristic:
                                                # we always set extra_heuristic = True, but
                                                # if extra_heuristic is not set, o may be sent before c
                                                if (o,c,r) not in self.is_before:
                                                    self.is_before[(o,c,r)] = opt.addVar(vtype=GRB.BINARY)
                                                is_before_ocr = self.is_before[(o,c,r)]
                                            else:
                                                assert (o,c,r) not in self.is_before
                                            if (c,o,r) not in self.is_before:
                                                self.is_before[(c,o,r)] = opt.addVar(vtype=GRB.BINARY)
                                            if (c1,o1,r) not in self.is_together:
                                                self.is_together[(c1,o1,r)] = opt.addVar(vtype=GRB.BINARY)
                                            opt.addLConstr(self.is_before[(c,o,r)] + self.is_together[(c1,o1,r)] + is_before_ocr == 1)
                                            # send chunk together with another only if the previous chunk between the two has been sent together
                                            if j-1>i:
                                                opt.addLConstr(self.is_together[(c1,o1,r)] <= self.is_together[(c2,prev_o2,r)])
                                        else:
                                            self.is_together_set_0.add((c1,o1,r))
                                            self.is_before_set_1.add((c,o,r))
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
                    if (o,src,r,l) in self.is_sent_set_1:
                        o1,c1 = minmax(o,c)
                        if (o1,c1,r) in self.is_together_set_1:
                            assert (o1,c1,r) not in self.is_together
                            num_s = num_s + 1
                            continue
                        if (o1,c1,r) in self.is_together_set_0:
                            assert (o1,c1,r) not in self.is_together
                        else:
                            if (o1,c1,r) not in self.is_together:
                                self.is_together[(o1,c1,r)] = opt.addVar(vtype=GRB.BINARY)
                lat = alpha(src,r) + beta(src,r)*(num_s + quicksum(self.is_together[(o,c,r)] if ((o,src,r,l) in self.is_sent_set_1 and (o,c,r) in self.is_together) else 0 for o in range(c)) + quicksum(self.is_together[(c,o,r)] if ((o,src,r,l) in self.is_sent_set_1 and (c,o,r) in self.is_together) else 0 for o in range(c,C)))
                return lat
            return alpha(src,r) + beta(src,r)

        # Set chunk is_send_set
        _add_chunk_sent(opt, heuristic)
        
        # Populate values
        for c in self.collective.chunks():
            for r in self.collective.ranks():
                recvd_anytime = [sum([1 if (c,src,r,l) in self.is_sent_set_1 else 0 for src in self.topology.sources(r)]) for l in range(L)]
                recv_IB = [sum([1 if ((c,src,r,l) in self.is_sent_set_1 and self._is_relay_link(src,r)) else 0 for src in self.topology.sources(r)]) for l in range(L)]
                if sum(recvd_anytime) == 0:
                    for l in range(L):
                        assert (c,c,r) not in self.is_together_set_1
                        assert (c,c,r) not in self.is_together
                        self.is_together_set_0.add((c,c,r))
                else:
                    # Will receive a chunk at most once
                    assert sum(recvd_anytime) == 1
                    # for l in range(L):
                    assert (c,c,r) not in self.is_together_set_1
                    assert (c,c,r) not in self.is_together
                        # if recvd_anytime[l] == 1:
                    self.is_together_set_1.add((c,c,r))

        # Set ordering
        recv_right_after = {}
        send_right_after = {}
        recv_right_after, send_right_after = _add_switch_order(
                                                switch_chunk_order_recv,
                                                switch_chunk_order_send,
                                                switch_link_mapping_recv,
                                                switch_link_mapping_send)
        _add_chunk_order(opt, heuristic, recv_right_after, send_right_after)

        def _get_isbefore(c,o,r):
            if (c,o,r) in self.is_before_set_1:
                return True, 1
            elif (c,o,r) in self.is_before:
                return False, self.is_before[(c,o,r)]
            else:
                return True, 0

        def _get_istogether(c,o,r):
            c1,o1 = minmax(c,o)
            if (c1,o1,r) in self.is_together_set_1:
                return True, 1
            elif (c1,o1,r,l) in self.is_together:
                return False, self.is_together[(c1,o1,r)]
            else:
                return True, 0

        # Correctness constraints
        for r in self.collective.ranks():
            src_r = [src for src in self.topology.sources(r)]
            links_r = {src: self.topology.link(src,r) for src in src_r}
            for c in self.collective.chunks():
                opt.addLConstr(self.start[c,r] <= ST)
                if r not in self.spsets[c]:
                    opt.addLConstr(self.start[c,r] == ST)
                    for src in src_r:
                        for l in range(L):
                            opt.addLConstr(self.send[c,src,r,l] == SND)
                    continue
                if self.collective.precondition(r, c):
                    opt.addLConstr(self.start[c,r] == 0)
                else:
                    # Bandwidth constraint
                    for src in src_r:
                        for l in range(links_r[src]):
                            if (c,src,r,l) in self.is_sent_set_1:
                                opt.addLConstr(self.start[c,r] == self.send[c,src,r,l] + calc_latency(src,r,l,c))
                            else:
                                opt.addLConstr(self.send[c,src,r,l] >= SND)
                        for l in range(links_r[src], L):
                            opt.addLConstr(self.send[c,src,r,l] == SND)
                    recvd_anytime = sum([sum([1 if (c,src,r,l) in self.is_sent_set_1 else 0 for l in range(links_r[src])]) for src in src_r])
                    if self.collective.postcondition(r, c):
                        opt.addLConstr(self.start[c,r] <= self.time)
                        assert recvd_anytime == 1, f'{c} {r} {self.is_sent_set_1}'
                    else:
                        assert recvd_anytime <= 1
                        if recvd_anytime == 0:
                            opt.addLConstr(self.start[c,r] >= self.time + 1)
                        else:
                            opt.addLConstr(self.start[c,r] <= self.time)

                for src in src_r:
                    for l in range(links_r[src]):
                        if (c,src,r,l) in self.is_sent_set_1:
                            opt.addLConstr(self.start[c,src] <= self.start[c,r])
                        opt.addLConstr(self.start[c,src] <= self.send[c,src,r,l])


                # Order sends from same gpu to same gpu
                for o in range(c):
                    is_static_cor, is_before_cor = _get_isbefore(c,o,r)
                    is_static_ocr, is_before_ocr = _get_isbefore(o,c,r)
                    is_static_t_ocr, is_together_ocr = _get_istogether(o,c,r)
                    # chunks sent together must have same send and start time
                    if is_static_t_ocr and is_together_ocr == 1:
                        for src in src_r:
                            for l in range(self.topology.link(src,r)):
                                if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1:
                                    opt.addLConstr(self.send[c,src,r,l] == self.send[o,src,r,l])
                        opt.addLConstr(self.start[c,r] == self.start[o,r])
                    elif not is_static_t_ocr:
                        for src in src_r:
                            for l in range(self.topology.link(src,r)):
                                if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1:
                                    opt.addGenConstrIndicator(self.is_together[(o,c,r)], True, self.send[c,src,r,l] == self.send[o,src,r,l])
                        opt.addGenConstrIndicator(self.is_together[(o,c,r)], True, self.start[c,r] == self.start[o,r])


                    if is_static_cor and is_static_ocr and is_static_t_ocr:
                        sent_same = any([1 if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1 else 0 for l in range(L) for src in self.topology.sources(r)])
                        sent_val = 1 if sent_same else 0
                        assert is_before_cor + is_before_ocr + is_together_ocr == sent_val, f'assertion error: {is_before_cor}, {is_before_ocr}, {is_together_ocr}, {sent_val}, {sent_same}, {c}, {o}, {r}, {l}'

                    # Bandwidth constraints based on chunk send times
                    for src in src_r:
                        for l in range(self.topology.link(src,r)):
                            if (c,src,r,l) in self.is_sent_set_1 and (o,src,r,l) in self.is_sent_set_1:
                                lat_o = calc_latency(src,r,l,o)
                                lat_c = calc_latency(src,r,l,c)

                                if (c,o,r) in self.is_before_set_1:
                                    opt.addLConstr(self.send[c,src,r,l] + lat_c <= self.send[o,src,r,l])
                                elif (c,o,r) in self.is_before:
                                    opt.addLConstr(self.send[c,src,r,l] + lat_c <= self.send[o,src,r,l] + M*(1-self.is_before[(c,o,r)]))
                                if (o,c,r) in self.is_before_set_1:
                                    opt.addLConstr(self.send[o,src,r,l] + lat_o <= self.send[c,src,r,l])
                                elif (o,c,r) in self.is_before:
                                    opt.addLConstr(self.send[o,src,r,l] + lat_o <= self.send[c,src,r,l] + M*(1-self.is_before[(o,c,r)]))

        num_local_nodes = R // self.topology.copies
        # Order receives from a switch
        for (c,src,r,l) in self.is_sent_set_1:
            if (src,r) in self.topology.switches_involved_in:
                for (swt_i, swt_type) in self.topology.switches_involved_in[(src,r)]:
                    srcs_check = []
                    if l == swt_i:
                        for srcs, dsts, _, _, switch_name in self.topology.switches[swt_i]:
                            if r in dsts and "in" in switch_name and src in srcs:
                                srcs_check = srcs
                                assert len(srcs_check)>0, f'{r} {c} {src} {l} {self.topology.switches[l]}'
                                break
                        lat_c = calc_latency(src,r,l,c)
                        for o in range(c+1):
                            for src_o in srcs_check:
                                if src_o == src:
                                    continue
                                if (o,src_o,r,l) in self.is_sent_set_1:
                                    if o == c:
                                        assert False
                                    lat_o = calc_latency(src_o,r,l,o)
                                    if (o,c,r,l,src_o,src) in self.recv_first_set_1:
                                        # opt.addLConstr(self.start[o,r] + lat_c <= self.start[c,r])
                                        opt.addLConstr(self.send[o,src_o,r,l] + lat_o <= self.send[c,src,r,l])
                                    elif (c,o,r,l,src,src_o) in self.recv_first_set_1:
                                        # opt.addLConstr(self.start[c,r] + lat_o <= self.start[o,r])
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
                for (swt_i, swt_type) in self.topology.switches_involved[(r,dst)]:
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


    def optimize(self, chunk_order=None, chunk_time=None,
                switch_chunk_order_recv=None, switch_chunk_time_recv=None,
                switch_chunk_order_send=None, switch_chunk_time_send=None, 
                nic_chunk_order_recv=None, nic_chunk_time_recv=None,
                nic_chunk_order_send=None, nic_chunk_time_send=None,
                switch_link_mapping_recv=None, switch_link_mapping_send=None):
        import math
        from time import time
        print(self.topology.name)
        chunkup = self.route_sketch.hyperparameters.chunkup
        print("chunkup =", chunkup)
        instance_name = 'taccl_{}_{}'.format(self.topology.name, self.collective.name)
        start_time = time()
        opt = Model(instance_name)
        self._encode(opt, chunk_order, chunk_time, 
            switch_chunk_order_recv, switch_chunk_time_recv,
            switch_chunk_order_send, switch_chunk_time_send,
            nic_chunk_order_recv, nic_chunk_time_recv,
            nic_chunk_order_send, nic_chunk_time_send,
            switch_link_mapping_recv, switch_link_mapping_send)
        # opt.write(f'model_{instance_name}.lp')
        opt.optimize()
        end_time = time()
        print("strict time (encode+solve)", end_time-start_time, flush=True)

        if opt.status == GRB.INFEASIBLE:
            opt.computeIIS()
            opt.write("model.ilp")
            raise ValueError("Infeasible model")

        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = self.topology.L

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
                        t = int(SCALE_TIME*self.send[c,src,r,l].X + 0.0001)
                        transfer_str = f'{c}: {src} --{l}--> {r}  t={self.send[c,src,r,l].X}\n'
                        recv_times[t].append(transfer_str)
                        send_dict[t].append([c,src,r,t,l])
                        chunk_path[c][t].append(transfer_str)
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
                    if (o,c,r) in self.is_together:
                        if self.is_together[(o,c,r)].X >= 0.995:
                            model_str += f'({c},{o},{r})\n'
                    elif (o,c,r) in self.is_together_set_1:
                        model_str += f'({c},{o},{r}) set\n'

        steps=[]
        send_times = sorted(send_dict.keys())
        i = 0
        while(i < len(send_times)):
            num_sends = [[[0 for _ in range(L)] for _ in range(R)] for _ in range(R)]
            j = i + 1
            while j < len(send_times):
                to_break = False
                t_end = send_times[j]
                for (c,src,r,_,l) in send_dict[t_end]:
                    for t in range(i,j):
                        for (ci,_,ri,_,li) in send_dict[send_times[t]]:
                            if c == ci and src == ri:
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
            num_sends = [[[0 for _ in range(L)] for _ in range(R)] for _ in range(R)]
            for (c,src,r,_,l) in sends:
                num_sends[r][src][l] = num_sends[r][src][l] + 1
            rounds = 0
            for srcs, dsts, bw, l, name in self.topology.real_bandwidth_constraints():
                util = 0
                for dst in dsts:
                    for src in srcs:
                        util += num_sends[dst][src][l]
                if rounds <= util * bw * SCALE_TIME:
                    rounds = math.ceil(util * bw * SCALE_TIME)
            steps.append(Step(rounds, sorted(sends, key=lambda x: x[3])))
            i = j

        instance = Instance(
            steps=len(steps),
            extra_rounds=0,
            chunks=chunkup,
        )
        soltype = "a" if chunk_order is None else "improve"
        from time import time
        timestamp = int(time())
        np.save(f'send_dict_{timestamp}.npy', send_dict)
        return Algorithm.make_implementation(self.collective, self.topology, instance, steps, cont=True, suffix=f'-tacclsol-{soltype}-{timestamp}')
