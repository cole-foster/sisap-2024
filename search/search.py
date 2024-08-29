'''
    Cole Foster
    August 4th, 2024

    SISAP 2024 Indexing Challenge
'''
import argparse
import GraphHierarchy
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time

# data_directory = "data"
data_directory = "/users/cfoste18/scratch/datasets/LAION"

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def prepare(kind, size):
    dataset_base_url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    task = {
        # "query": "http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
        "query": f"{dataset_base_url}/public-queries-10k-clip768v2.h5",
        "dataset": f"{dataset_base_url}/laion2B-en-{kind}-n={size}.h5"
    }

    # print(f"okay did I prepared?")

    for version, url in task.items():
        download(url, os.path.join(data_directory, kind, size, f"{version}.h5"))

def store_results(dst, algo, kind, D, I, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()


def run(size, max_neighbors, s, p, search_neighbors):
    kind = "clip768v2"
    key = "emb"
    index_identifier = f"AHSP-M-{max_neighbors}-s-{s}-p-{p}"
    print(f"Running {index_identifier} on {kind}-{size}")
    if (search_neighbors <= 0):
        search_neighbors = max_neighbors
    
    #> Download dataset if necessary
    prepare(kind, size)
    D=768

    #> Initialize the HNSW index
    index = GraphHierarchy.Index(space='ip', dim=D) # possible options are l2, cosine or ip

    #> Load the dataset
    start_time = time.time()
    with h5py.File(os.path.join(data_directory, kind, size, "dataset.h5"), 'r') as f:
        dataset = f[key]
        N,DD = dataset.shape
        print(f'Datset has N={N} rows and D={DD} columns')
        index.init_index(dataset_size=N, max_neighbors=max_neighbors, random_seed=0)
        print(" * Initiated index")
        
        # determine number of rows
        total_rows = dataset.shape[0]
        chunk_size = 100000

        # iterate over the dataset, add each chunk
        for start_index in range(0, total_rows, chunk_size):
            end_index = min(start_index + chunk_size, total_rows)

            # load this chunk into memory, add to index
            data_chunk = dataset[start_index:end_index]
            index.add_items(data_chunk)

    print(f" * done adding items {time.time() - start_time:.4} (s)")

    # construct the bottom layer graph
    index.build(s, p)
    index.construct_partitioning(search_neighbors) # num neighbors per node... consistent
    build_time = time.time() - start_time
    print(f"Done Constructing Index in {build_time:.4f} (s)")

    # get the queries
    queries = np.array(h5py.File(os.path.join(data_directory, kind, size, "query.h5"), "r")[key],dtype=np.float32)

    index.set_search_neighbors(search_neighbors)

    #> Searching on the index
    beam_size_vec = [30, 35, 40, 45, 50, 55, 60, 70, 85, 100, 120, 150, 200, 300, 400, 500, 650, 800, 1000]
    for beam_size in beam_size_vec:
        print(f"Searching with beam_size={beam_size}")
        start = time.time()
        index.set_beam_size(beam_size)  # ef should always be > k
        labels, distances = index.search(queries, k=30)
        search_time = time.time() - start
        print(f"Done searching in {search_time:.4}s.")

        # # save the results
        labels = labels + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        identifier = f"index=({index_identifier}),query=(b={beam_size})"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), index_identifier, kind, distances, labels, build_time, search_time, identifier, size)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=str,
        default="300K"
    )
    parser.add_argument(
        "-M",
        type=int,
        default=16
    )
    parser.add_argument(
        "-s",
        type=int,
        default=10
    )
    parser.add_argument(
        "-p",
        type=int,
        default=100
    )
    parser.add_argument(
        "-m",
        type=int,
        default=0
    )
    args = parser.parse_args()
    assert args.size in ["300K", "10M", "100M"]

    print("Running Script With:")
    print(f"  * N={args.size}")
    print(f"  * M={args.M}")
    print(f"  * s={args.s}")
    print(f"  * p={args.p}")
    print(f"  * m={args.m}")
    run(args.size, args.M, args.s, args.p, args.m)
    print(f"Done! Have a good day!")
