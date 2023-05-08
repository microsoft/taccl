
# [dgx2-allgather-sk1-n2]: 2-node NDVIDIA DGX-2 AllGather for 1MB data chunks using sketch-1
taccl solve DGX2 Allgather --topology-file ../taccl/examples/topo/topo-dgx2-1MB.json --sketch-file ../taccl/examples/sketch/sk1-dgx2-n2.json

# [dgx2-allgather-sk2-n2]: 2-node NDVIDIA DGX-2 AllGather for 1KB data chunks using sketch-
taccl solve DGX2 Allgather --topology-file ../taccl/examples/topo/topo-dgx2-1KB.json --sketch-file ../taccl/examples/sketch/sk2-dgx2-n2.json

# [ndv2-allgather-sk1-n2]: 2-node Azure NDv2 AllGather for 1MB data chunks using sketch-1
taccl solve NDv2 Allgather --topology-file ../taccl/examples/topo/topo-ndv2-1MB.json --sketch-file ../taccl/examples/sketch/sk1-ndv2-n2.json

# [dgx2-alltoall-sk2-n2]: 2-node NVIDIA DGX-2 AlltoAll for 1KB data chunks using sketch-2
taccl solve DGX2 Alltoall --topology-file ../taccl/examples/topo/topo-dgx2-1KB.json --sketch-file ../taccl/examples/sketch/sk2-dgx2-n2.json

# dgx2-alltoall-sk3-n2
taccl solve DGX2 Alltoall --topology-file ../taccl/examples/topo/topo-dgx2-1KB.json --sketch-file ../taccl/examples/sketch/sk3-dgx2-n2.json

# ndv2-alltoall-sk1-n2
taccl solve NDv2 Alltoall --topology-file ../taccl/examples/topo/topo-ndv2-1MB.json --sketch-file ../taccl/examples/sketch/sk1-ndv2-n2.json

# ndv2-alltoall-sk2-n2
taccl solve NDv2 Alltoall --topology-file ../taccl/examples/topo/topo-ndv2-1KB.json --sketch-file ../taccl/examples/sketch/sk2-ndv2-n2.json

# dgx2-allreduce-sk1-n2
taccl combine DGX2 Allgather --topology-file ../taccl/examples/topo/topo-dgx2-1MB.json --sketch-file ../taccl/examples/sketch/sk1-dgx2-n2.json --ts <ts>

# dgx2-allreduce-sk2-n2
taccl combine DGX2 Allgather --topology-file ../taccl/examples/topo/topo-dgx2-1KB.json --sketch-file ../taccl/examples/sketch/sk2-dgx2-n2.json --ts <ts>

# ndv2-allreduce-sk1-n2
taccl combine NDv2 Allgather --topology-file ../taccl/examples/topo/topo-ndv2-1MB.json --sketch-file ../taccl/examples/sketch/sk1-ndv2-n2.json --ts <ts>


# ndv2-allgather-sk1-n4
taccl combine NDv2 Allgather --topology-file ../taccl/examples/topo/topo-ndv2-1MB.json --sketch-file ../taccl/examples/sketch/sk1-ndv2-n4.json

# ndv2-allgather-sk1-n6
taccl solve DGX1 Allgather --topology-file ../taccl/examples/topo/topo-ndv2-1MB.json --sketch-file ../taccl/examples/sketch/sk1-ndv2-n6.json

# ndv2-allgather-sk1-n8
taccl solve DGX1 Allgather --topology-file ../taccl/examples/topo/topo-ndv2-1MB.json --sketch-file ../taccl/examples/sketch/sk1-ndv2-n8.json

