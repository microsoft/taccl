# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
import copy
from time import time
import numpy as np

class HeuristicOrderer(object):
    def __init__(self, topology, route_sketch, collective, reverse=False):
        self.topology = topology
        self.route_sketch = route_sketch
        self.collective = collective
        self.heuristic = -1
        self.paths = None
        self.reverse = reverse
        assert "DGX2Single" not in self.topology.name

    # Returns a list of the reverse path traversed by a chunk
    # chunk c:
    #   0 -> 1 -> 2
    #   0 -> 1 -> 3
    # output path: paths[c] = [ [(1,2,bw), (0,1,bw)], [(1,3,bw), (0,1,bw)] ]
    def set_paths(self, chunk_send):
        paths = defaultdict(list)
        def has_next(c,r):
            for dst in self.topology.destinations(r):
                for l in range(self.topology.link(r,dst)):
                    if c in chunk_send[r][dst][l]:
                        return True
            return False

        def prev(c,r):
            if self.collective.precondition(r,c):
                return -1, -1
            for src in self.topology.sources(r):
                for l in range(self.topology.link(src,r)):
                    if c in chunk_send[src][r][l]:
                        return src, l
            return -2, -2

        for r in range(self.collective.num_nodes):
            for src in self.topology.sources(r):
                for l in range(self.topology.link(src,r)):
                    for c in chunk_send[src][r][l]:
                        # trace back chunk path from the end-destinations of the chunk
                        # Assumption: a rank does not receive the same chunk from two different sources
                        if not has_next(c,r):
                            path = [(src,r,self.topology.get_invbw(src,r),l)]
                            p_r = src
                            while True:
                                p_src, p_l = prev(c,p_r)
                                if p_src == -1:
                                    break
                                elif p_src == -2:
                                    assert False
                                else:
                                    path.append((p_src,p_r,self.topology.get_invbw(p_src,p_r),p_l))
                                    p_r = p_src
                            paths[c].append(path)
        self.paths = paths
        return paths

    def latency(self, src, dst, l):
        return self.topology.get_invbw(src,dst)

    # returns 0 if own node chunk, and 1 if not
    def is_own_node_chunk(self,c,src,r):
        num_local_nodes = self.num_local_nodes
        pre_nodes = []
        for r1 in self.collective.pre_rank(c):
            pre_nodes.append(r1//num_local_nodes)
        if src // num_local_nodes == r // num_local_nodes and src // num_local_nodes in pre_nodes:
            return 0
        else:
            return 1

    # Main function to calculate heuristics from paths
    # returns pos_last[src,r,l, reverse index of segment in path] = (scheduled time, chunk, pathid)
    def get_last_pos(self, paths, to_travel_list, to_travel_sum_list, to_travel_branch_sum_list, has_travelled_list, R, heuristic, L):
        path_labels = None
        num_local_nodes = R // self.topology.copies
        if self.reverse:
            assert (heuristic == 12) or (heuristic==18)
        algtime = 0
        def dst_new_pos(new_pos,src_next,r_next,l,ii,pi,c):
            for (t_dst, c_dst, p_dst) in new_pos[(src_next,r_next,l,ii)]:
                if p_dst == pi:
                    assert c_dst == c
                    return t_dst - self.latency(src_next,r_next,l)
            assert False, "shouldn't reach here"
        pos_last = defaultdict(list)
        new_pos_last = defaultdict(list)
        max_len = 0
        for c in paths:
            for path in paths[c]:
                if len(path) >= max_len:
                    max_len = len(path)
        path_list = []  # list of all paths
        chunk_list = [] # list of chunks corresponding to the paths
        path_bw_list = []   # time for uninterrupted flow of chunk on the paths
        for c in paths:
            for i in range(len(paths[c])):
                path_list.append(paths[c][i])
                chunk_list.append(c)
                path_bw_list.append(sum([bw for (_,_,bw,_) in paths[c][i]]))
        selected_cand_heuristic = [6,8,9,10,11,12,18,13,14,15]
        if heuristic not in selected_cand_heuristic:
            # Schedule chunks from the last send in the path
            # Sort paths with the first being the one with longest hops in the path
            path_list, chunk_list, path_bw_list = zip(*sorted(zip(path_list, chunk_list, path_bw_list), key=lambda x: -len(x[0])))

            for i in range(max_len):
                # Start from the n-th send for each path
                # Schedule the last sends at each path to happen at time = algtime
                # pos_last[src,r,l, reverse index of chunk]
                for j, path in enumerate(path_list):
                    if len(path) > i:
                        src,r,bw,l = path[i]
                        chunk = chunk_list[j]
                        if i == 0:
                            pos_last[(src,r,l,i)].append((algtime, chunk, j))
                        else:
                            pos_last[(src,r,l,i)].append((-1, chunk, j))
                    else:
                        continue
                for (srci,ri,l,j) in pos_last:
                    if j == i:
                        lat = self.latency(srci,ri,l)
                        if i == 0:
                            # schedule that first (and hence last) that has to travel the least, has travelled the max, and in a round robin manner after that
                            for k, (t,c,p) in enumerate(sorted(pos_last[(srci,ri,l,i)], key=lambda x: (to_travel_list[(x[1],srci,ri)], -has_travelled_list[(x[1],srci,ri)], (srci%num_local_nodes-ri%num_local_nodes+num_local_nodes)%num_local_nodes, ri))):
                                new_pos_last[(srci,ri,l,i)].append((t - k*lat, c, p))
                        else:
                            t_curr = 0
                            ii = i-1
                            while ii >= 0:
                                if (srci,ri,l,ii) in new_pos_last:
                                    for (t_prev,_,_) in new_pos_last[(srci,ri,l,ii)]:
                                        if t_prev <= t_curr:
                                            t_curr = t_prev
                                ii = ii - 1
                            # for the path segments which have the same to_travel and has_travelled, consider the path first which has most dst_new_pos
                            # thus, you should consider the one that can be scheduled the first from the end
                            for k, (t,c,p) in enumerate(sorted(pos_last[(srci,ri,l,i)], key=lambda x: (to_travel_list[(x[1],srci,ri)], -has_travelled_list[(x[1],srci,ri)], 
                                                                -dst_new_pos(new_pos_last, path_list[p][i-1][0], path_list[p][i-1][1], path_list[p][i-1][3], i-1, p, chunk_list[p]),
                                                                (srci%num_local_nodes-ri%num_local_nodes+num_local_nodes)%num_local_nodes, ri))):
                                np_last = new_pos_last[(path_list[p][i-1][0],path_list[p][i-1][1],path_list[p][i-1][3],i-1)]
                                has_next = False
                                for (t_next, c_next, p_next) in np_last:
                                    if c_next == c and p_next == p:
                                        t_prev = t_next - self.latency( path_list[p_next][i-1][0],path_list[p_next][i-1][1],l)
                                        t_curr = min(t_curr-self.latency(srci,ri,l), t_prev)
                                        new_pos_last[(srci,ri,l,i)].append((t_curr, c, p))
                                        has_next = True
                                assert has_next
        elif heuristic == 10 or heuristic == 11 or heuristic == 13 or heuristic == 14 or heuristic == 15:
            # Sort paths with the first being the one with longest total time in the path
            path_list, chunk_list, path_bw_list = zip(*sorted(zip(path_list, chunk_list, path_bw_list), key=lambda x: -x[2]))

            # reverse the path - makes it correct order of sends
            # we will schedule the chunk segments for each path from the start
            path_list_new = [p[::-1] for p in path_list]
            path_list = path_list_new

            algtime = 0
            ptime = [algtime] * len(path_list)  # denote time of path
            ltime = [[[algtime for _ in range(L)] for _ in range(R)] for _ in range(R)] # h10_2 # denotes time of link

            # pointer to the segment in each path
            pptr = [0 for p in range(len(path_list))]
            num_local_nodes = R // self.topology.copies

            selected_cands = []     # [(p,src,r,bw,l,c),...]
            # from all paths, select a segment which is pointed to by pptr
            for i in range(len(path_list)):
                if pptr[i] != -1:
                    segment = path_list[i][pptr[i]]
                    selected_cands.append((i,segment[0],segment[1],segment[2],segment[3],chunk_list[i]))

            def is_input(c,src):
                if self.collective.precondition(src,c):
                    return 1
                else:
                    return 0

            # time that segment can be placed
            def cand_time(p,src,r,l,verbose=False):
                cand_ptime = ptime[p]
                cand_ltime = ltime[src][r][l]
                if verbose:
                    print ("cand_time", cand_ptime, cand_ltime)
                return max(cand_ptime, cand_ltime)

            # repeat until we run out of segments to place
            while len(selected_cands):
                if heuristic == 10:
                    # h10c
                    # first send that segment which has to travel the most. For segments that have to travel the same distance, send the one first which can be sent first (cand_time)
                    # if same cand time, then send the one with smaller difference between send and receive pairs (just a heuristic),
                    # that with lower receiver local rank, that with lower sender local rank, that with lower receiver node, and finally that with the highest path length
                    sorted_selected_cands = sorted(selected_cands, key=lambda x: (-to_travel_list[(x[5],x[1],x[2])], cand_time(x[0],x[1],x[2],x[4]), (x[1]-x[2]+R)%R, x[2]%num_local_nodes, x[1]%num_local_nodes, x[2]//num_local_nodes, x[0]))
                elif heuristic == 13:
                    # h13: choose to send chunks of your own node first if both have same to_travel time and cand_time
                    # lambda key: [(p,src,r,bw,l,c),...]
                    sorted_selected_cands = sorted(selected_cands, key=lambda x: (-to_travel_list[(x[5],x[1],x[2])], cand_time(x[0],x[1],x[2],x[4]), self.is_own_node_chunk(x[5],x[1],x[2]), (x[1]-x[2]+R)%R, x[2]%num_local_nodes, x[1]%num_local_nodes, x[2]//num_local_nodes, x[0]))
                elif heuristic == 14:
                    # h14: changes order of heuristics
                    # lambda key: [(p,src,r,bw,l,c),...]
                    sorted_selected_cands = sorted(selected_cands, key=lambda x: (cand_time(x[0],x[1],x[2],x[4]), self.is_own_node_chunk(x[5],x[1],x[2]), -to_travel_list[(x[5],x[1],x[2])],  (x[1]-x[2]+R)%R, x[2]%num_local_nodes, x[1]%num_local_nodes, x[2]//num_local_nodes, x[0]))
                    x = sorted_selected_cands[0]
                    print("---", x, cand_time(x[0],x[1],x[2],x[4]))
                elif heuristic == 15:
                    sorted_selected_cands = sorted(selected_cands, key=lambda x: (cand_time(x[0],x[1],x[2],x[4]), -to_travel_list[(x[5],x[1],x[2])],  (x[1]-x[2]+R)%R, x[2]%num_local_nodes, x[1]%num_local_nodes, x[2]//num_local_nodes, x[0]))
                    x = sorted_selected_cands[0]
                    print("---", x, cand_time(x[0],x[1],x[2],x[4]))
                else:
                    sorted_selected_cands = sorted(selected_cands, key=lambda x: (-to_travel_list[(x[5],x[1],x[2])], cand_time(x[0],x[1],x[2],x[4]), (x[1]-x[2]+R)%R, -is_input(x[4],x[1]), x[2]%num_local_nodes, x[1]%num_local_nodes, x[2]//num_local_nodes, x[0]))
                # schedule the first segment from the sorted candidates
                selected_one = sorted_selected_cands[0] # (pathid, src,r,bw,l,c)
                p, src, r, _, l,  c  = selected_one
                # get time to schedule the segment
                time = cand_time(p,src,r,l)
                lat = self.latency(src,r,l)
                time_new = time + lat
                new_pos_last[(src,r,l,0)].append((time,c,p))
                chosen_cands = []
                chosen_cands.append((src,r,l,c))
                if (src,r) in self.topology.switches_involved:
                    for (swt_i, swt_type) in self.topology.switches_involved[(src,r)]:
                        if swt_i == l:
                            # update the link time for each switch link if we scheduled a segment on one switch link
                            for srcs, dsts, _, _, switch_name in self.topology.switches[swt_i]:
                                if r in dsts and "in" in switch_name:
                                    for srci in srcs:
                                        ltime[srci][r][swt_i] = time_new
                                if src in srcs and "out" in switch_name:
                                    for ri in dsts:
                                        ltime[src][ri][swt_i] = time_new
                # update the ltime of the segment.
                ltime[src][r][l] = time_new
                # Also update the ptimes of all paths with which the segment is involved in. Update their pptr as well
                # Invariant - Segments which are common for different paths will appear together as selected_cands because their has_travelled time is the same
                # Invariant -  the selected_one will always have the same path time as all other p_common because they will have the same paths till now
                for j, sscand in enumerate(sorted_selected_cands):
                    if (sscand[1], sscand[2], sscand[4], sscand[5]) in chosen_cands:
                        p_common = sscand[0]
                        # the path time is update
                        ptime[p_common] = time_new
                        if pptr[p_common] == len(path_list[p_common]) - 1:
                            pptr[p_common] = -1
                            sorted_selected_cands.remove(sscand)
                        else:
                            pptr[p_common] = pptr[p_common] + 1
                            common_segment = path_list[p_common][pptr[p_common]]
                            sorted_selected_cands[j] = (p_common,common_segment[0],common_segment[1],common_segment[2],common_segment[3],chunk_list[p_common])
                selected_cands = sorted_selected_cands
            max_time = 0
            for src in range(R):
                for r in range(R):
                    for l in range(L):
                        if max_time <= ltime[src][r][l]:
                            max_time = ltime[src][r][l]
            for p in range(len(path_list)):
                if max_time <= ptime[p]:
                    max_time = ptime[p]
        elif heuristic == 12 or heuristic == 18:
            # used for reversed allgather
            assert self.reverse
            path_list, chunk_list, path_bw_list = zip(*sorted(zip(path_list, chunk_list, path_bw_list), key=lambda x: -x[2]))
            path_list_new = [p[::-1] for p in path_list]
            path_list = path_list_new
            algtime = 0
            ptime = [algtime] * len(path_list)
            ltime = [[[algtime for _ in range(L)] for _ in range(R)] for _ in range(R)] # h10_2
            pptr = [0 for p in range(len(path_list))]
            num_local_nodes = R // self.topology.copies

            selected_cands = []     # [(p,src,r,bw,l,c),...]
            # from all paths, select a segment which is pointed to by pptr
            for i in range(len(path_list)):
                if pptr[i] != -1:
                    segment = path_list[i][pptr[i]]
                    selected_cands.append((i,segment[0],segment[1],segment[2],segment[3],chunk_list[i]))

            def cand_time(p,src,r,l, verbose=False):
                cand_ptime = ptime[p]
                cand_ltime = ltime[src][r][l]
                if verbose:
                    print ("cand_time", cand_ptime, cand_ltime)
                return max(cand_ptime, cand_ltime)

            while len(selected_cands):
                # h12c
                if heuristic == 12:
                    sorted_selected_cands = sorted(selected_cands, key=lambda x: (-to_travel_sum_list[(x[5],x[1],x[2])], cand_time(x[0],x[1],x[2],x[4]), (x[1]-x[2]+R)%R, x[2]%num_local_nodes, x[1]%num_local_nodes, x[2]//num_local_nodes, x[0]))
                else:
                    sorted_selected_cands = sorted(selected_cands, key=lambda x: (-to_travel_branch_sum_list[(x[5],x[1],x[2])], cand_time(x[0],x[1],x[2],x[4]), (x[1]-x[2]+R)%R, x[2]%num_local_nodes, x[1]%num_local_nodes, x[2]//num_local_nodes, x[0]))
                # schedule the first segment from the sorted candidates
                selected_one = sorted_selected_cands[0]
                p, src, r, _, l, c  = selected_one
                # get time to schedule the segment
                time = cand_time(p,src,r,l)
                lat = self.latency(src,r,l)
                time_new = time + lat
                new_pos_last[(src,r,l,0)].append((time,c,p))
                chosen_cands = []
                chosen_cands.append((src,r,l,c))
                if (src,r) in self.topology.switches_involved:
                    for (swt_i, swt_type) in self.topology.switches_involved[(src,r)]:
                        if swt_i == l:
                            for srcs, dsts, _, _, switch_name in self.topology.switches[swt_i]:
                                if r in dsts and "in" in switch_name:
                                    for srci in srcs:
                                        ltime[srci][r][swt_i] = time_new
                                if src in srcs and "out" in switch_name:
                                    for ri in dsts:
                                        ltime[src][ri][swt_i] = time_new
                # update the ltime of the segment.
                ltime[src][r][l] = time_new
                # Also update the ptimes of all paths with which the segment is involved in. Update their pptr as well
                # If the path has two branches, update ptime of the untravelled branches but don't update their pptr
                # Invariant - Segments which are common for different paths will appear together as selected_cands because their has_travelled time is the same
                for j, sscand in enumerate(sorted_selected_cands):
                    if (sscand[1], sscand[2], sscand[4], sscand[5]) in chosen_cands:
                        p_common = sscand[0]
                        ptime[p_common] = time_new
                        if pptr[p_common] == len(path_list[p_common]) - 1:
                            pptr[p_common] = -1
                            sorted_selected_cands.remove(sscand)
                        else:
                            pptr[p_common] = pptr[p_common] + 1
                            next_segment = path_list[p_common][pptr[p_common]]
                            sorted_selected_cands[j] = (p_common,next_segment[0],next_segment[1],next_segment[2],next_segment[3],chunk_list[p_common])
                    else:
                        for (c_src,c_r,c_l,c_c) in chosen_cands:
                            # a chunk reduce is non-atomic, so it must be done one after other
                            if sscand[1] == c_src and sscand[4] == c_l and sscand[5] == c_c:
                                p_branch = sscand[0]
                                ptime[p_branch] = time_new

                selected_cands = sorted_selected_cands
        return new_pos_last, path_labels

    def get_metadata(self, paths):
        # get a list of the reverse path traversed by a chunk
        # paths = self.set_paths(chunk_send)
        self.paths = paths


        # the time to perform all branches in the path,
        # if each branch from a node has to happen after another branch has been started
        to_travel_branch_sum_list = {}
        
        def to_travel_branch_sum(c,src,dst):
            assert self.paths is not None
            if (c,src,dst) in to_travel_branch_sum_list:
                return to_travel_branch_sum_list[(c,src,dst)]
            to_travel_bw_branch_sum = 0
            num_branches = 0
            branch_list = []
            this_segment_bw = 0
            for pathid, path in enumerate(self.paths[c]):
                to_travel_bw = 0
                path_bw = 0
                path_goes_through = False
                for (srci,dsti,bw,l) in path:
                    if srci == src and dsti == dst:
                        path_goes_through = True
                        break
                if path_goes_through:
                    to_travel_sum_prior = 0
                    for (srci,dsti,bw,l) in path:
                        if srci == src and dsti == dst:
                            # print(f'Added {c},{src},{dst} to to_travel_branch, path id: {pathid}')
                            branch_list.append(to_travel_sum_prior)
                            this_segment_bw = bw
                            break
                        else:
                            # assert (c,srci,dsti) in to_travel_branch_sum_list, f'({c},{srci},{dsti}), ({c},{src},{dst}) {path}'
                            to_travel_sum_prior = to_travel_branch_sum(c,srci,dsti)
            
            assert this_segment_bw != 0
            assert len(branch_list) > 0
            assert (c,src,dst) not in to_travel_branch_sum_list
            # print(c,src,dst, this_segment_bw, branch_list)
            to_travel_branch_sum_list[(c,src,dst)] = this_segment_bw + max(branch_list)
            return to_travel_branch_sum_list[(c,src,dst)]

        # time c has to travel from src to dst and further summed for each dsts coming out from dst
        def to_travel_sum(c,src,dst):
            assert self.paths is not None
            to_travel_bw_sum = 0
            num_branches = 0
            for path in self.paths[c]:
                to_travel_bw = 0
                path_bw = 0
                for (srci,dsti,bw,l) in path:
                    path_bw = path_bw + self.latency(srci,dsti,l)
                    if srci == src and dsti == dst:
                        to_travel_bw += path_bw
                        break
                to_travel_bw_sum += to_travel_bw
            if to_travel_bw_sum == 0:
                assert False, f'missing {c} in {src} -> {dst}'
            else:
                return to_travel_bw_sum

        # time c has to travel from src to dst and further
        def to_travel(c,src,dst):
            assert self.paths is not None
            to_travel_bw_max = 0
            for path in self.paths[c]:
                to_travel_bw = 0
                path_bw = 0
                for (srci,dsti,bw,l) in path:
                    path_bw = path_bw + self.latency(srci,dsti,l)
                    if srci == src and dsti == dst:
                        to_travel_bw += path_bw
                        break
                to_travel_bw_max = max(to_travel_bw, to_travel_bw_max)
            if to_travel_bw_max == 0:
                assert False, f'missing {c} in {src} -> {dst}'
            else:
                return to_travel_bw_max

        # time c has travelled till reaching src
        def has_travelled(c,src,dst):
            assert self.paths is not None
            has_travelled_bw = 0
            has_travelled_flag = False
            for path in paths[c]:
                path_bw = 0
                for (srci,dsti,bw,l) in reversed(path):
                    if srci == src and dsti == dst:
                        has_travelled_bw += path_bw
                        has_travelled_flag = True
                        break
                    path_bw = path_bw + self.latency(srci,dsti,l)
            if not has_travelled_flag:
                assert has_travelled_bw == 0
                assert False, f'missing {c} in {src} -> {dst}'
            else:
                return has_travelled_bw

        p_list = []
        to_travel_list = {}
        to_travel_sum_list = {}
        has_travelled_list = {}
        for c in paths:
            for i in range(len(paths[c])):
                p_list.append((c,paths[c][i]))
                for (src,r,_,_) in paths[c][i]:
                    to_travel_list[(c,src,r)] = to_travel(c,src,r)
                    to_travel_sum_list[(c,src,r)] = to_travel_sum(c,src,r)
                    has_travelled_list[(c,src,r)] = has_travelled(c,src,r)
        for c in paths:
            for path in paths[c]:
                (src,r,_,_) = path[-1]
                to_travel_branch_sum(c,src,r)
        return paths, to_travel_list, to_travel_sum_list, to_travel_branch_sum_list, has_travelled_list

    def perform_ordering(self, chunk_send, time_send, chunk_recv, time_recv, distribute_over_links=True):
        # TODO add distribute over links
        # For now, does not explicitly distribute over links
        chunkup = self.route_sketch.hyperparameters.chunkup
        heuristic = self.route_sketch.hyperparameters.heuristic
        distribute_over_links == False

        C = self.collective.num_chunks
        R = self.collective.num_nodes
        L = self.topology.L

        LL = 0
        if self.topology.copies > 1:
            switch_set = [defaultdict(set) for i in range(2)]
            for i, switch in enumerate(self.topology.switches):
                for srcs, dsts, _, _, switch_name in switch:
                    if "out" in switch_name:
                        assert len(srcs) == 1
                        rank = srcs[0]
                        switch_set_idx = 0
                    else:
                        assert len(dsts) == 1
                        rank = dsts[0]
                        switch_set_idx = 1
                    # print(rank, swtic)
                    switch_set[switch_set_idx][rank].add(switch_name)
                    LL = max(LL, len(switch_set[switch_set_idx][rank]))
        else:
            LL = L

        num_local_nodes = R // self.topology.copies
        self.num_local_nodes = num_local_nodes

        heur_start_time = time()

        # From the chunk_send, time_send, chunk_recv, time_recv determine 
        switch_link_recv = [[None for l in range(LL)] for r in range(R)]
        switch_time_recv = [[[] for l in range(LL)] for r in range(R)]
        switch_chunk_recv = [[[] for l in range(LL)] for r in range(R)]
        switch_link_send = [[None for l in range(LL)] for r in range(R)]
        switch_time_send = [[[] for l in range(LL)] for r in range(R)]
        switch_chunk_send = [[[] for l in range(LL)] for r in range(R)]
        # switch_map_rev[r][(src,l)] = ll: map a transfer from src --> r over link l to a switch link ll
        self.topology.switch_map_rev = [defaultdict() for r in range(R)]
        # switch_map_rev[r][(dst,l)] = ll: map a transfer from r --> dst over link l to a switch link ll
        self.topology.switch_map_send = [defaultdict() for r in range(R)]

        paths = self.set_paths(chunk_send)
        paths, to_travel_list, to_travel_sum_list, to_travel_branch_sum_list, has_travelled_list = self.get_metadata(paths)
        pos_last, path_labels = self.get_last_pos(paths, to_travel_list, to_travel_sum_list, to_travel_branch_sum_list, has_travelled_list,R,heuristic,L)

        # for c in sorted(paths):
        #     for path in paths[c]:
        #         print("path", c, len(path))

        def order_pos_last(c, src, r, l):
            for (srci,ri,li,i) in pos_last:
                if srci == src and ri == r and li == l:
                    for t, ci, _ in pos_last[src,r,l,i]:
                        if c == ci:
                            return t
            assert False, f'no order_pos found {c},{src},{r}'


        for r1 in range(R): # dst for recv
            ll = 0
            sw_added = []
            for r2 in range(R): # source for recv
                if (r2,r1) in self.topology.switches_involved_in:
                    l = 0
                    # Assumes all r2 to r1 connections have same l
                    for (swt_i, swt_type) in self.topology.switches_involved_in[(r2,r1)]:
                        if (swt_i, swt_type) not in sw_added:
                            l = swt_i
                            for srcs, dsts, _, _, switch_name in self.topology.switches[swt_i]:
                                if r1 in dsts and "in" in switch_name and r2 in srcs:
                                    assert swt_type in switch_name
                                    # print("switch: r,srcs",r1,srcs,l,ll)
                                    switch_link_recv[r1][ll] = l
                                    # print(r1, ll)
                                    for src in srcs:
                                        self.topology.switch_map_rev[r1][(src,l)] = ll
                                        if len(time_recv[r1][src][l]):
                                            switch_time_recv[r1][ll].extend(zip(time_recv[r1][src][l], [src]*len(time_recv[r1][src][l])))
                                            switch_chunk_recv[r1][ll].extend(zip(chunk_recv[r1][src][l], [src]*len(chunk_recv[r1][src][l])))
                                    if len(switch_chunk_recv[r1][ll]):
                                        if heuristic == 14 or heuristic == 15:
                                            # Do not consider the original times output by path encoding, instead, rely on the order_pos_lat computed by heuristics
                                            switch_time_recv[r1][ll], switch_chunk_recv[r1][ll] = zip(*sorted(zip(switch_time_recv[r1][ll], switch_chunk_recv[r1][ll]), key=lambda x: (order_pos_last(x[1][0],x[0][1],r1,l), (x[0][1]-r1 + R)%R)))
                                        else:
                                            switch_time_recv[r1][ll], switch_chunk_recv[r1][ll] = zip(*sorted(zip(switch_time_recv[r1][ll], switch_chunk_recv[r1][ll]), key=lambda x: (x[0][0], order_pos_last(x[1][0],x[0][1],r1,l), (x[0][1]-r1 + R)%R)))
                                        # print("switch_chunk_recv", r1)
                                        # print(switch_time_recv[r1][ll])
                                        for i in range(len(switch_chunk_recv[r1][ll])):
                                            c_recv = switch_chunk_recv[r1][ll][i][0]
                                            s_recv = switch_chunk_recv[r1][ll][i][1]
                                            t_recv = switch_time_recv[r1][ll][i][0]
                                            l_recv = switch_link_recv[r1][ll]
                                            # p_recv = path_labels[(s_recv,r1,l_recv,c_recv)]
                                            assert l_recv == l
                                            # tup = (i, s_recv, c_recv, t_recv,
                                            #  self.is_own_node_chunk(c_recv,s_recv,r1),
                                            #  -to_travel_list[(c_recv,s_recv,r1)],
                                            #  order_pos_last(c_recv,s_recv,r1,l_recv),(s_recv-r1 + R)%R)
                                            # print(tup, end=",")
                                        # print("")
                            ll = ll + 1
                            sw_added.append((swt_i, swt_type))

        for r1 in range(R): # source for send
            ll = 0
            sw_added = []
            for r2 in range(R): # dst for send
                if (r1,r2) in self.topology.switches_involved:
                    l = 0
                    # Assumes all r1 to r2 connections have same l
                    for (swt_i, swt_type) in self.topology.switches_involved[(r1,r2)]:
                        if (swt_i, swt_type) not in sw_added:
                            l = swt_i
                            for srcs, dsts, _, _, switch_name in self.topology.switches[swt_i]:
                                if r1 in srcs and "out" in switch_name and r2 in dsts:
                                    assert swt_type in switch_name
                                    switch_link_send[r1][ll] = l
                                    for dst in dsts:
                                        self.topology.switch_map_send[r1][(dst,l)] = ll
                                        if len(time_send[r1][dst][l]):
                                            switch_time_send[r1][ll].extend(zip(time_send[r1][dst][l], [dst]*len(time_send[r1][dst][l])))
                                            switch_chunk_send[r1][ll].extend(zip(chunk_send[r1][dst][l], [dst]*len(chunk_send[r1][dst][l])))
                                    # TODO need a heuristic which ensures that nic group transfers are ordered together
                                    if len(switch_chunk_send[r1][ll]):
                                        if heuristic == 14 or heuristic == 15:
                                            switch_time_send[r1][ll], switch_chunk_send[r1][ll] = zip(*sorted(zip(switch_time_send[r1][ll], switch_chunk_send[r1][ll]), key=lambda x: (order_pos_last(x[1][0],r1,x[0][1],l), (r1-x[0][1]+R)%R)))
                                        else:
                                            switch_time_send[r1][ll], switch_chunk_send[r1][ll] = zip(*sorted(zip(switch_time_send[r1][ll], switch_chunk_send[r1][ll]), key=lambda x: (x[0][0], order_pos_last(x[1][0],r1,x[0][1],l), (r1-x[0][1]+R)%R)))

                                        # print("switch_chunk_send", r1, *zip(list(range(len(switch_chunk_send[r1][ll]))),switch_time_send[r1][ll], switch_chunk_send[r1][ll]))
                                    # l = l + 1
                                    # break
                            ll = ll + 1
                            sw_added.append((swt_i, swt_type))

        # for r1 in range(R):
        #     print("switch_map_rev:", r1, self.topology.switch_map_rev[r1])
        # for r1 in range(R):
        #     print("switch_map_send:", r1, self.topology.switch_map_send[r1])
        for r1 in range(R): # source
            for r2 in range(R): #dst
                # if (r1,r2) not in self.topology.switches_involved:
                for l in range(L):
                    # Sort chunks and times for each rank and link
                    if (len(time_send[r1][r2][l])):
                        # r1: source, r2: dst
                        if heuristic == 14 or heuristic == 15:
                            time_send[r1][r2][l], chunk_send[r1][r2][l] = zip(*sorted(zip(time_send[r1][r2][l], chunk_send[r1][r2][l]), key=lambda x: (order_pos_last(x[1],r1,r2,l), (r1-r2 + R)%R)))
                        else:
                            time_send[r1][r2][l], chunk_send[r1][r2][l] = zip(*sorted(zip(time_send[r1][r2][l], chunk_send[r1][r2][l]), key=lambda x: (x[0], order_pos_last(x[1],r1,r2,l), (r1-r2 + R)%R)))
                        # print("chunk_send", r1, r2, *zip(list(range(len(chunk_send[r1][r2][l]))),chunk_send[r1][r2][l]))
                    if (len(time_recv[r2][r1][l])):
                        # r1: source, r2: dst
                        if heuristic == 14 or heuristic == 15:
                            time_recv[r2][r1][l], chunk_recv[r2][r1][l] = zip(*sorted(zip(time_recv[r2][r1][l], chunk_recv[r2][r1][l]), key=lambda x: (order_pos_last(x[1],r1,r2,l), (r1-r2 + R)%R)))
                        else:
                            time_recv[r2][r1][l], chunk_recv[r2][r1][l] = zip(*sorted(zip(time_recv[r2][r1][l], chunk_recv[r2][r1][l]), key=lambda x: (x[0], order_pos_last(x[1],r1,r2,l), (r1-r2 + R)%R)))
                        # print("chunk_recv", r2, r1, *zip(list(range(len(chunk_recv[r2][r1][l]))),chunk_recv[r2][r1][l]))
                    chunk_send[r1][r2][l] = np.array(chunk_send[r1][r2][l])
                    chunk_recv[r2][r1][l] = np.array(chunk_recv[r2][r1][l])
                    time_send[r1][r2][l] = np.array(time_send[r1][r2][l])
                    time_recv[r2][r1][l] = np.array(time_recv[r2][r1][l])

        if self.reverse:
            chunk_recv_rev = [[[None for l in range(L)] for r2 in range(R)] for r1 in range(R)]
            time_recv_rev = [[[None for l in range(L)] for r2 in range(R)] for r1 in range(R)]
            chunk_send_rev = [[[None for l in range(L)] for r2 in range(R)] for r1 in range(R)]
            time_send_rev = [[[None for l in range(L)] for r2 in range(R)] for r1 in range(R)]
            switch_time_recv_rev = [[[] for l in range(LL)] for r in range(R)]
            switch_chunk_recv_rev = [[[] for l in range(LL)] for r in range(R)]
            switch_time_send_rev = [[[] for l in range(LL)] for r in range(R)]
            switch_chunk_send_rev = [[[] for l in range(LL)] for r in range(R)]
            for r1 in range(R):
                for ll in range(len(switch_chunk_recv[r1])):
                    for i in range(len(switch_chunk_recv[r1][ll])-1,-1,-1):
                        switch_chunk_send_rev[r1][ll].append(switch_chunk_recv[r1][ll][i])
                        switch_time_send_rev[r1][ll].append((-switch_time_recv[r1][ll][i][0],switch_time_recv[r1][ll][i][1]))
                for ll in range(len(switch_chunk_send[r1])):
                    for i in range(len(switch_chunk_send[r1][ll])-1,-1,-1):
                        switch_chunk_recv_rev[r1][ll].append(switch_chunk_send[r1][ll][i])
                        switch_time_recv_rev[r1][ll].append((-switch_time_send[r1][ll][i][0],switch_time_send[r1][ll][i][1]))
            for r1 in range(R):
                for r2 in range(R):
                    for l in range(len(chunk_send[r1][r2])):
                        chunk_recv_rev[r1][r2][l] = np.flipud(chunk_send[r1][r2][l])
                        time_recv_rev[r1][r2][l] = -np.flipud(time_send[r1][r2][l])
                    for l in range(len(chunk_recv[r1][r2])):
                        chunk_send_rev[r1][r2][l] = np.flipud(chunk_recv[r1][r2][l])
                        time_send_rev[r1][r2][l] = -np.flipud(time_recv[r1][r2][l])
            return [time_recv_rev, chunk_recv_rev, switch_time_recv_rev, switch_chunk_recv_rev, switch_time_send_rev, switch_chunk_send_rev, None, None, None, None, switch_link_recv, switch_link_send, paths]

        heur_end_time = time()
        print("heurtime {}\n ".format(heur_end_time-heur_start_time))
        return [time_recv, chunk_recv, switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, None, None, None, None, switch_link_recv, switch_link_send, paths]
