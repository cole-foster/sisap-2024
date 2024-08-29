#pragma once
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "distances.h"
#include "utils.hpp"
#include "visited_list_pool.h"

typedef unsigned uint;
struct Level {
    size_t num_pivots_{0};
    std::vector<uint> pivots_{};
    std::unordered_map<uint, uint> map_{};
    std::vector<std::vector<uint>> partitions_{};
    std::vector<std::vector<uint>> graph_{};

    void add_pivot(uint const pivot) {
        size_t idx = num_pivots_;
        num_pivots_++;
        pivots_.push_back(pivot);
        map_[pivot] = idx;
    }
    void initialize_partitions() { partitions_.resize(num_pivots_); }
    void initialize_graph() { graph_.resize(num_pivots_); }
    void set_neighbors(uint const pivot, std::vector<uint> const& neighbors) { graph_[map_.at(pivot)] = neighbors; }
    void add_member(uint const pivot, uint const member) { partitions_[map_.at(pivot)].push_back(member); }
    std::vector<uint> const& get_neighbors(uint pivot) const { return graph_[map_.at(pivot)]; }
    std::vector<uint> const& get_partition(uint pivot) const { return partitions_[map_.at(pivot)]; }
};

class GraphHierarchy {
   public:

    // data: store the graph and data together
    char* data_bottom_level_{nullptr};
    const uint dataset_size_ = 0;
    const size_t max_neighbors_{32};
    size_t size_links_{0};        // memory for graph (per node): uint + max_neighbors_*uint
    size_t size_data_{0};         // memory for data (per node): dimension * sizeof(float)
    size_t size_per_element_{0};  // memory for graph and data (per node)

    // distances
    distances::SpaceInterface<float>* space_;
    DISTFUNC<float> distFunc_;
    void* distFuncParam_{nullptr};

    // hierarchy
    int random_seed_ = 100;
    std::vector<Level> hierarchy_{};
    int num_levels_{0};
    std::default_random_engine level_generator_;

    // graph construction parameters
    int beamSizeConstruction_ = 10;

    // search parameters
    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};
    int search_beam_size_{10};
    int search_max_neighbors_{0};

    GraphHierarchy() {};

    GraphHierarchy(uint dataset_size, distances::SpaceInterface<float>* s, uint max_neighbors=32): 
                dataset_size_(dataset_size), max_neighbors_((size_t)max_neighbors) {
        printf("Initialized GraphHierarchy Object:\n");
        printf(" * N=%u\n", dataset_size_);
        printf(" * max_neighbors=%u\n", max_neighbors);

        // using the distance function from hnswlib
        space_ = s;
        distFunc_ = s->get_dist_func();
        distFuncParam_ = s->get_dist_func_param();
        size_data_ =  s->get_data_size();      

        // from hnswlib: store graph and data together
        size_links_ = sizeof(uint) + max_neighbors_ * sizeof(uint);
        size_per_element_ = size_links_ + size_data_;
        data_bottom_level_ = (char *)malloc(dataset_size_ * size_per_element_);
        if (data_bottom_level_ == nullptr) throw std::runtime_error("Not enough memory");

        // initializing the visited list object for multithreading
        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, dataset_size_));

        srand(3);
    }
    ~GraphHierarchy() {
        visited_list_pool_.reset(nullptr);
        free(data_bottom_level_);
        data_bottom_level_ = nullptr;
    };

    // stats
    std::chrono::high_resolution_clock::time_point tStart, tEnd;
    std::chrono::high_resolution_clock::time_point tStart1, tEnd1;
    std::chrono::high_resolution_clock::time_point tStart2, tEnd2;

    // distance computations
    char* getDataByIndex(uint index) const { 
        return (data_bottom_level_ + index * size_per_element_ + size_links_); // data is offset by links
    }
    float compute_distance(char* index1_ptr, char* index2_ptr) const {
        return distFunc_(index1_ptr, index2_ptr, distFuncParam_);
    }
    float compute_distance(char* index1_ptr, uint index2) const {
        return distFunc_(index1_ptr, getDataByIndex(index2), distFuncParam_);
    }
    float compute_distance(uint index1, char* index2_ptr) const {
        return distFunc_(getDataByIndex(index1), index2_ptr, distFuncParam_);
    }
    float compute_distance(uint index1, uint index2) const {
        return distFunc_(getDataByIndex(index1), getDataByIndex(index2), distFuncParam_);
    }

    // add each data point and initialize the bottom layer graph
    void addPoint(const void* element_ptr, uint element) {
        if (element >= dataset_size_) {
            throw std::runtime_error("Element ID exceeded dataset_size_");
        }
        // - initializing and setting data memory for bottom level
        memset(data_bottom_level_ + element * size_per_element_, 0, size_per_element_); // initialize all mem
        memcpy(data_bottom_level_ + element * size_per_element_ + size_links_, element_ptr, size_data_); // set the data
        return;
    }

/*
|=======================================================================================================================
||
||
||                          GRAPH
||
||
|=======================================================================================================================
*/
    // - fetching the linked list holding the neighbors of an element
    uint* get_linklist(uint index) const {
        return (uint*) (data_bottom_level_ + index*size_per_element_);
    } 
    uint get_linklist_count(uint* ptr) const {
        return *(ptr);
    }
    void set_linklist_count(uint* ptr, uint count) const {
        *(ptr) = count;
    }
    void set_neighbors(uint index, std::vector<uint>& neighbors) const {
        uint num_neighbors = (uint) neighbors.size();
        if (num_neighbors > max_neighbors_) num_neighbors = max_neighbors_;

        uint* index_data = get_linklist(index);
        set_linklist_count(index_data, num_neighbors);
        
        uint* index_neighbors = (uint*)(index_data + 1);
        for (uint i = 0; i < num_neighbors; i++) {
            index_neighbors[i] = neighbors[i];
        }
    }
    uint* get_neighbors(uint index, uint& num_neighbors) const {
        uint* index_data = get_linklist(index);
        num_neighbors = get_linklist_count(index_data);
        return (uint*)(index_data + 1);
    }
    uint get_num_neighbors(uint index) const {
        uint* index_data = get_linklist(index);
        return get_linklist_count(index_data);
    }

    void saveGraph(std::string filename, uint dimension) {
        std::vector<std::vector<uint>> graph(dataset_size_);
        for (uint index = 0; index < dataset_size_; index++) {
            uint num_neighbors;
            uint* neighbors = get_neighbors(index, num_neighbors);
            graph[index].resize(num_neighbors);
            for (uint i = 0; i < num_neighbors; i++) {
                graph[index][i] = neighbors[i];
            }
        }
        printf("Saving graph to: %s\n", filename.c_str());
        Utils::saveGraph(filename, graph, dimension, dataset_size_);
        return;
    }
    void loadGraph(std::string filename, uint dimension) {
        printf("Loading graph from: %s\n", filename.c_str());

        // get the graph
        std::vector<std::vector<uint>> graph(dataset_size_);
        Utils::loadGraph(filename, graph, dimension, dataset_size_);

        // set the graph
        for (uint index = 0; index < dataset_size_; index++) {
            if (graph[index].size() > max_neighbors_) graph[index].resize(max_neighbors_);
            set_neighbors(index, graph[index]);
        }
        printGraphStats();
    }

    void printGraphStats() {
        printf("Graph Statistics:\n");
        printf(" * number of nodes: %u\n", dataset_size_);

        // - get the average degree
        size_t min_degree = 1000000;
        size_t max_degree = 0;
        size_t num_edges = 0.0f;
        for (uint node = 0; node < dataset_size_; node++) {
            size_t node_edges = get_num_neighbors(node);
            num_edges += node_edges;
            if (node_edges < min_degree) min_degree = node_edges;
            if (node_edges > max_degree) max_degree = node_edges;
        }
        printf(" * number of edges: %lu\n", num_edges);
        double ave_degree = (double)num_edges / (double)dataset_size_;
        printf(" * average degree: %.2f\n", ave_degree);
        printf(" * min degree: %d\n", (int)min_degree);
        printf(" * max degree: %d\n", (int)max_degree);
        printf(" * (ave/min/max): %.2f, %d, %d\n", ave_degree, min_degree, max_degree);
        fflush(stdout);
        return;
    }

    /*
    |=======================================================================================================================
    ||
    ||
    ||                          CONSTRUCTION
    ||
    ||
    |=======================================================================================================================
    */

    // construct the exact hsp graph
    void constructExactHSPGraph(uint maxK = 0) {
        printf("Begin HSP Graph Construction\n");
        printf(" * N = %u\n", dataset_size_);
        printf(" * maxNeighbors: %u\n", max_neighbors_);
        if (maxK == 0) {
            printf(" * Exact HSP!\n");
        } else {
            printf(" * kNN-Constrained HSP with k=%u\n", maxK);
        }
        fflush(stdout);

        std::vector<uint> dataset_vector(dataset_size_);
        std::iota(dataset_vector.begin(), dataset_vector.end(), 0);

        // Apply HSP test to each node in parallel
        tStart = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (uint node = 0; node < dataset_size_; node++) {
            std::vector<uint> neighbors = HSPTest(node, dataset_vector, maxK);
            set_neighbors(node, neighbors);
        }
        tEnd = std::chrono::high_resolution_clock::now();
        double time_graph = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
        printf(" * Graph Construction Time (s): %.4f\n", time_graph);
        fflush(stdout);
        printGraphStats();
    }

    // perform the hsp test to get the hsp neighbors of the node
    std::vector<uint> HSPTest(uint const query, std::vector<uint> const& set, int maxK = 0) const {
        std::vector<uint> neighbors{};
        char* queryPtr = getDataByIndex(query);

        // - initialize the active list A
        std::vector<std::pair<float, uint>> active_list{};
        active_list.reserve(set.size());

        // - initialize the list with all points and distances, find nearest neighbor
        uint index1;
        float distance_Q1 = HUGE_VAL;
        for (uint index : set) {
            if (index == query) continue;
            float const distance = compute_distance(queryPtr, getDataByIndex(index));
            if (distance < distance_Q1) {
                distance_Q1 = distance;
                index1 = index;
            }
            active_list.emplace_back(distance, index);
        }

        // - limit the number of points to consider
        if (maxK > 0 && active_list.size() > maxK) {
            // - nth_element sort: bring the kth closest elements to the front, but not sorted, O(N)
            std::nth_element(active_list.begin(), active_list.begin() + maxK, active_list.end());
            active_list.resize(maxK);  // keep the top k points
        }

        // - perform the hsp loop
        while (active_list.size() > 0) {
            if (max_neighbors_ > 0 && neighbors.size() >= max_neighbors_) break;

            // - next neighbor as closest valid point
            neighbors.push_back(index1);
            char* index1_ptr = getDataByIndex(index1);

            // - set up for the next hsp neighbor
            uint index1_next;
            float distance_Q1_next = HUGE_VAL;

            // - initialize the active_list for next iteration
            // - make new list: push_back O(1) faster than deletion O(N)
            std::vector<std::pair<float, uint>> active_list_copy = active_list;
            active_list.clear();

            // - check each point for elimination
            for (int it2 = 0; it2 < (int)active_list_copy.size(); it2++) {
                uint const index2 = active_list_copy[it2].second;
                float const distance_Q2 = active_list_copy[it2].first;
                if (index2 == index1) continue;
                float const distance_12 = compute_distance(index1_ptr, getDataByIndex(index2));

                // - check the hsp inequalities: add if not satisfied
                if (distance_Q1 >= distance_Q2 || distance_12 >= distance_Q2) {
                    active_list.emplace_back(distance_Q2, index2);

                    // - update neighbor for next iteration
                    if (distance_Q2 < distance_Q1_next) {
                        distance_Q1_next = distance_Q2;
                        index1_next = index2;
                    }
                }
            }

            // - setup the next hsp neighbor
            index1 = index1_next;
            distance_Q1 = distance_Q1_next;
        }

        return neighbors;
    }

    /*
    |=======================================================================================================================
    ||
    ||
    ||                          HIERARCHICAL PARTITIONING
    ||
    ||
    |=======================================================================================================================
    */

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    // initialize the hierarchy
    void initializeHierarchy(int const scaling, int num_levels = 0) {
        hierarchy_.clear();

        // estimate the number of levels
        num_levels_ = num_levels;
        if (num_levels_ <= 0) {
            double start_value = (double)dataset_size_;
            while (start_value >= 10) {
                num_levels_++;
                start_value /= (double)scaling;
            }
            hierarchy_.resize(num_levels_ - 1);
        }
        hierarchy_.resize(num_levels_ - 1);
        printf(" * Number of levels: %d\n", num_levels_);

        // select pivots probabilistically on each level, same as hnsw
        double mult = 1 / log((double)scaling);
        level_generator_.seed(random_seed_);
        for (uint index = 0; index < dataset_size_; index++) {
            int level_assignment = getRandomLevel(mult);  // 0 as bottom, anything greater is top
            if (level_assignment > num_levels_ - 1) level_assignment = num_levels_ - 1;

            // add to each level
            // want top to be level 0, bottom to be level num_levels_-1
            for (int l = 1; l <= level_assignment; l++) {
                int level_of_interest = num_levels_ - l - 1;
                hierarchy_[level_of_interest].add_pivot(index);
            }
        }
        for (int l = 0; l < hierarchy_.size(); l++) {
            printf("    - %d --> %u\n", l, (uint)hierarchy_[l].num_pivots_);
        }
        printf("    - %d --> %u (bottom)\n", num_levels_ - 1, dataset_size_);

        fflush(stdout);
        return;
    }

    // Hierarchical partitioning for search time
    void constructPartitioning(int const scaling) {
        printf("|> Creating hierarchical partitioning...\n");
        fflush(stdout);
        tStart = std::chrono::high_resolution_clock::now();

        // Initilaize the hierarchy
        initializeHierarchy(scaling);

        //> Construct the Partitioning on Each Level
        for (int ell = 1; ell < num_levels_ - 1; ell++) {
            printf(" * Begin level-%d partitioning...\n", ell);
            tStart1 = std::chrono::high_resolution_clock::now();

            // - initializations
            const size_t num_pivots = hierarchy_[ell].num_pivots_;
            printf("    - num_pivots = %u\n", (uint)num_pivots);
            fflush(stdout);
            hierarchy_[ell - 1].initialize_partitions();
            std::vector<uint> pivot_assignments(num_pivots);

            //> Perform the partitioning of this current layer
            // - perform this task using all threads
            #pragma omp parallel for
            for (size_t itp = 0; itp < num_pivots; itp++) {
                const uint fine_pivot = hierarchy_[ell].pivots_[itp];
                char* fine_pivot_ptr = getDataByIndex(fine_pivot);
                uint closest_pivot;

                // - top-down assignment to a coarse pivot
                for (int c = 0; c < ell; c++) {
                    // - define the candidate pivots in the layer
                    std::vector<uint> candidate_coarse_pivots{};
                    if (c == 0) {
                        candidate_coarse_pivots = hierarchy_[c].pivots_;
                    } else {
                        candidate_coarse_pivots = hierarchy_[c - 1].get_partition(closest_pivot);
                    }

                    // - find and record the closest coarse pivot
                    float closest_dist = HUGE_VAL;
                    for (uint coarse_pivot : candidate_coarse_pivots) {
                        float dist = compute_distance(fine_pivot_ptr, coarse_pivot);
                        if (dist < closest_dist) {
                            closest_dist = dist;
                            closest_pivot = coarse_pivot;
                        }
                    }
                }

                // - record the closest coarse pivot found (thread safe)
                pivot_assignments[itp] = closest_pivot;
            }
            // - assign to the partitions (thread safe)
            for (size_t itp = 0; itp < num_pivots; itp++) {
                uint const fine_pivot = hierarchy_[ell].pivots_[itp];
                hierarchy_[ell - 1].add_member(pivot_assignments[itp], fine_pivot);
            }
            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_level = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf("    - level time (s): %.4f\n", time_level);
            fflush(stdout);
        }

        //> Perform the partitioning of the bottom level
        {
            int ell = num_levels_ - 1;
            printf(" * Begin level-%d partitioning... (bottom level)\n", ell);
            printf("    - num elements = %u\n", (uint)dataset_size_);
            fflush(stdout);
            tStart1 = std::chrono::high_resolution_clock::now();

            // - initializations
            hierarchy_[ell - 1].initialize_partitions();
            std::vector<uint> pivot_assignments(dataset_size_);

            #pragma omp parallel for
            for (uint node = 0; node < dataset_size_; node++) {
                char* node_ptr = getDataByIndex(node);
                uint closest_pivot;

                // - top-down assignment to a coarse pivot
                for (int c = 0; c < ell; c++) {
                    // - define the candidate pivots in the layer
                    std::vector<uint> candidate_coarse_pivots{};
                    if (c == 0) {
                        candidate_coarse_pivots = hierarchy_[c].pivots_;
                    } else {
                        candidate_coarse_pivots = hierarchy_[c - 1].get_partition(closest_pivot);
                    }

                    // - find and record the closest coarse pivot
                    float closest_dist = HUGE_VAL;
                    for (uint coarse_pivot : candidate_coarse_pivots) {
                        float const dist = compute_distance(node_ptr, coarse_pivot);
                        if (dist < closest_dist) {
                            closest_dist = dist;
                            closest_pivot = coarse_pivot;
                        }
                    }
                }

                // - record the closest coarse pivot found (thread safe)
                pivot_assignments[node] = closest_pivot;
            }
            // - assign to the partitions (thread safe)
            for (uint node = 0; node < dataset_size_; node++) {
                hierarchy_[ell - 1].add_member(pivot_assignments[node], node);
            }
            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_level = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf("    - level time (s): %.4f\n", time_level);
            fflush(stdout);
        }
        tEnd = std::chrono::high_resolution_clock::now();
        double time_partition = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
        printf(" * Total Hierarchical Partitioning Time (s): %.4f\n", time_partition);
        fflush(stdout);
        return;
    }

/*
|=======================================================================================================================
||
||
||                          APPROXIMATE GRAPH CONSTRUCTION
||
||
|=======================================================================================================================
*/

    // graph parameters
    void buildGraph(int scaling = 100, int b = 40) {
        printf("Begin Index Construction...\n");
        tStart = std::chrono::high_resolution_clock::now();
        std::vector<uint> (GraphHierarchy::*edge_func)(uint, std::vector<uint> const&); // for different methods
        edge_func = &GraphHierarchy::edges_HSP;

        //> Initialize Hierarchy
        hierarchy_.clear();
        initializeHierarchy(scaling);

        //> Top-Down Construction, level-by-level
        printf("Begin Index Construction...\n");
        for (int ell = 1; ell < num_levels_ - 1; ell++) {
            printf(" * Begin level-%d construction...\n", ell);
            tStart1 = std::chrono::high_resolution_clock::now();

            // - initializations
            const size_t num_pivots = hierarchy_[ell].num_pivots_;
            printf("    - num_pivots = %u\n", (uint)num_pivots);
            fflush(stdout);
            hierarchy_[ell - 1].initialize_partitions();

            // graph needed
            bool flag_graph_needed = false;
            if (num_pivots * scaling >= 20000) flag_graph_needed = true;
            bool flag_coarse_graph = false;
            if (hierarchy_[ell - 1].graph_.size() > 0) flag_coarse_graph = true;

            // only perform the partitioning
            if (!flag_graph_needed) {
                printf("    - sufficiently small level. No graph needed!\n");
                printf("    - performing partitioning by brute force...\n");

                //> Perform the partitioning of this current layer
                tStart2 = std::chrono::high_resolution_clock::now();
                std::vector<uint> pivot_assignments(num_pivots);
                #pragma omp parallel for
                for (size_t itp = 0; itp < num_pivots; itp++) {
                    uint pivot = hierarchy_[ell].pivots_[itp];
                    char* pivot_ptr = getDataByIndex(pivot);

                    // find closest partition
                    uint closest_pivot;
                    float closest_dist = HUGE_VAL;
                    for (uint coarse_pivot : hierarchy_[ell - 1].pivots_) {
                        float dist = compute_distance(pivot_ptr, coarse_pivot);
                        if (dist < closest_dist) {
                            closest_dist = dist;
                            closest_pivot = coarse_pivot;
                        }
                    }

                    // - record the closest coarse pivot found (thread safe)
                    pivot_assignments[itp] = closest_pivot;
                }
                // - assign to the partitions (thread safe)
                for (size_t itp = 0; itp < num_pivots; itp++) {
                    uint pivot = hierarchy_[ell].pivots_[itp];
                    hierarchy_[ell - 1].add_member(pivot_assignments[itp], pivot);
                }
                tEnd2 = std::chrono::high_resolution_clock::now();
                double time_partition = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd2 - tStart2).count();
                printf("    - level-%d partition time (s): %.4f\n", ell, time_partition);
                tEnd1 = std::chrono::high_resolution_clock::now();
                double time_level = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
                printf("    - total level time (s): %.4f\n", time_level);
                fflush(stdout);
                continue;
            }

            printf("    - creating graph and partitioning in tandem\n");
            hierarchy_[ell].initialize_graph();

            //> Find the closest coarse pivots to the pivot
            tStart2 = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<uint>> vec_closest_coarse_pivots(num_pivots);
            #pragma omp parallel for
            for (size_t itp = 0; itp < num_pivots; itp++) {
                uint pivot = hierarchy_[ell].pivots_[itp];
                char* pivot_ptr = getDataByIndex(pivot);
                uint closest_coarse_pivot;  // to find, for assignment

                // - define the candidate list of neighbors
                if (!flag_coarse_graph) {  // brute force
                    float closest_coarse_dist = HUGE_VAL;
                    for (uint coarse_pivot : hierarchy_[ell - 1].pivots_) {
                        float dist = compute_distance(pivot_ptr, coarse_pivot);
                        if (dist < closest_coarse_dist) {
                            closest_coarse_dist = dist;
                            closest_coarse_pivot = coarse_pivot;
                        }
                    }
                    vec_closest_coarse_pivots[itp] = {closest_coarse_pivot};
                } else {  // by graph

                    // - top-down to get estimate closest pivot, starting point
                    for (int c = 0; c < ell; c++) {
                        // - collect the set of candidate coarse pivots in level c
                        std::vector<uint> candidate_coarse_pivots{};
                        if (c == 0) {
                            candidate_coarse_pivots = hierarchy_[c].pivots_;
                        } else {
                            candidate_coarse_pivots = hierarchy_[c - 1].get_partition(closest_coarse_pivot);
                        }

                        // - get the estimated closest pivot in level c
                        float closest_coarse_dist = HUGE_VAL;
                        for (uint coarse_pivot : candidate_coarse_pivots) {
                            float dist = compute_distance(pivot_ptr, coarse_pivot);
                            if (dist < closest_coarse_dist) {
                                closest_coarse_dist = dist;
                                closest_coarse_pivot = coarse_pivot;
                            }
                        }
                    }

                    // - Find the b closest nodes in level ell-1
                    vec_closest_coarse_pivots[itp] =
                        beamSearchConstruction(ell - 1, pivot_ptr, b, closest_coarse_pivot);
                }
            }
            // - assign to the partitions (thread safe)
            for (size_t itp = 0; itp < num_pivots; itp++) {
                uint const pivot = hierarchy_[ell].pivots_[itp];
                hierarchy_[ell - 1].add_member(vec_closest_coarse_pivots[itp][0], pivot);
            }
            tEnd2 = std::chrono::high_resolution_clock::now();
            double time_partition = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd2 - tStart2).count();
            printf("    - level-%d partition time (s): %.4f\n", ell, time_partition);

            // - now perform the graph construction
            tStart2 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (size_t itp = 0; itp < num_pivots; itp++) {
                uint pivot = hierarchy_[ell].pivots_[itp];

                // - define the candidate region
                std::vector<uint> candidate_region{};
                if (!flag_coarse_graph) {
                    candidate_region = hierarchy_[ell].pivots_;
                } else {
                    for (uint coarse_pivot : vec_closest_coarse_pivots[itp]) {
                        std::vector<uint> const& coarse_partition = hierarchy_[ell - 1].get_partition(coarse_pivot);
                        candidate_region.insert(candidate_region.end(), coarse_partition.begin(),
                                                coarse_partition.end());
                    }
                }

                // - perform the edge selection test on this neighborhood
                std::vector<uint> neighbors = (this->*edge_func)(pivot, candidate_region);
                hierarchy_[ell].set_neighbors(pivot, neighbors);
            }
            tEnd2 = std::chrono::high_resolution_clock::now();
            double time_graph = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd2 - tStart2).count();
            printf("    - level-%d graph time (s): %.4f\n", ell, time_graph);
            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_level = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf("    - total level time (s): %.4f\n", time_level);
            fflush(stdout);
        }

        //> Construct the Bottom Level Graph
        {
            int ell = num_levels_ - 1;
            printf(" * Begin level-%d construction... (bottom level)\n", ell);
            tStart1 = std::chrono::high_resolution_clock::now();

            // - initializations
            printf("    - num elements = %u\n", (uint)dataset_size_);
            fflush(stdout);

            // graph needed
            printf("    - creating graph and partitioning in tandem\n");
            bool flag_coarse_graph = false;
            if (hierarchy_[ell - 1].graph_.size() > 0) flag_coarse_graph = true;
            hierarchy_[ell - 1].initialize_partitions();

            //> Find the closest observers
            tStart2 = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<uint>> vec_closest_coarse_pivots(dataset_size_);
            #pragma omp parallel for
            for (uint node = 0; node < dataset_size_; node++) {
                char* node_ptr = getDataByIndex(node);

                // - define the candidate list of neighbors
                if (!flag_coarse_graph) {  // brute force

                    uint closest_coarse_pivot;
                    float closest_coarse_dist = HUGE_VAL;
                    for (uint coarse_pivot : hierarchy_[ell - 1].pivots_) {
                        float dist = compute_distance(node_ptr, coarse_pivot);
                        if (dist < closest_coarse_dist) {
                            closest_coarse_dist = dist;
                            closest_coarse_pivot = coarse_pivot;
                        }
                    }
                    vec_closest_coarse_pivots[node] = {closest_coarse_pivot};
                } else {  // by graph

                    // - top-down to get estimate closest pivot, starting point
                    uint closest_coarse_pivot;
                    for (int c = 0; c < ell - 1; c++) {
                        // - collect the set of candidate coarse pivots in level c
                        std::vector<uint> candidate_coarse_pivots{};
                        if (c == 0) {
                            candidate_coarse_pivots = hierarchy_[c].pivots_;
                        } else {
                            candidate_coarse_pivots = hierarchy_[c - 1].get_partition(closest_coarse_pivot);
                        }

                        // - get the estimated closest pivot in level c
                        float closest_coarse_dist = HUGE_VAL;
                        for (uint coarse_pivot : candidate_coarse_pivots) {
                            float dist = compute_distance(node_ptr, coarse_pivot);
                            if (dist < closest_coarse_dist) {
                                closest_coarse_dist = dist;
                                closest_coarse_pivot = coarse_pivot;
                            }
                        }
                    }

                    // - Find the b closest nodes in level ell-1
                    vec_closest_coarse_pivots[node] =
                        beamSearchConstruction(ell - 1, node_ptr, b, closest_coarse_pivot);
                }
            }
            // - assign to the partitions (thread safe)
            for (uint node = 0; node < dataset_size_; node++) {
                hierarchy_[ell - 1].add_member(vec_closest_coarse_pivots[node][0], node);
            }
            tEnd2 = std::chrono::high_resolution_clock::now();
            double time_partition = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd2 - tStart2).count();
            printf("    - level-%d partition time (s): %.4f\n", ell, time_partition);
            fflush(stdout);

            // - now perform the graph construction
            tStart2 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (uint node = 0; node < dataset_size_; node++) {
                char* node_ptr = getDataByIndex(node);

                // - define the candidate region
                std::vector<uint> candidate_region{};
                if (!flag_coarse_graph) {
                    candidate_region = hierarchy_[ell].pivots_;
                } else {
                    for (uint coarse_pivot : vec_closest_coarse_pivots[node]) {
                        std::vector<uint> const& coarse_partition = hierarchy_[ell - 1].get_partition(coarse_pivot);
                        candidate_region.insert(candidate_region.end(), coarse_partition.begin(),
                                                coarse_partition.end());
                    }
                }

                // - perform the edge selection test on this neighborhood
                std::vector<uint> neighbors = (this->*edge_func)(node, candidate_region);
                set_neighbors(node, neighbors);
            }
            tEnd2 = std::chrono::high_resolution_clock::now();
            double time_graph = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd2 - tStart2).count();
            printf("    - level-%d graph time (s): %.4f\n", ell, time_graph);

            tEnd1 = std::chrono::high_resolution_clock::now();
            double time_level = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
            printf("    - total level time (s): %.4f\n", time_level);
            fflush(stdout);
        }

        tEnd = std::chrono::high_resolution_clock::now();
        double time_index = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
        printf("    - total index time (s): %.4f\n", time_index);
        fflush(stdout);
        printGraphStats();
        return;
    }

    // performing a beam search to obtain the closest partitions for approximate hsp
    std::vector<uint> beamSearchConstruction(int level, char* query_ptr, int k, uint start_node) {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // initialize lists
        int beam_size = beamSizeConstruction_;
        if (beam_size < k) beam_size = k;
        std::priority_queue<std::pair<float, uint>> topCandidates;
        std::priority_queue<std::pair<float, uint>> candidateSet;

        float dist = compute_distance(query_ptr, start_node);
        topCandidates.emplace(dist, start_node);
        float lowerBound = dist;
        candidateSet.emplace(-dist, start_node);
        visited_array[start_node] = visited_array_tag;

        // perform the beam search
        while (!candidateSet.empty()) {
            std::pair<float, uint> current_pair = candidateSet.top();
            if ((-current_pair.first) > lowerBound && topCandidates.size() == beam_size) {
                break;
            }
            candidateSet.pop();

            // - fetch neighbors of current node
            uint const current_node = current_pair.second;
            std::vector<uint> const& current_node_neighbors = hierarchy_[level].get_neighbors(current_node);
            size_t num_neighbors = current_node_neighbors.size();

            // - iterate through the neighbors
            for (size_t j = 0; j < num_neighbors; j++) {
                uint neighbor_node = current_node_neighbors[j];

                // - skip if already visisted
                if (visited_array[neighbor_node] == visited_array_tag) continue;
                visited_array[neighbor_node] = visited_array_tag;

                // - update data structures if applicable
                float dist = compute_distance(query_ptr, neighbor_node);
                if (topCandidates.size() < beam_size || lowerBound > dist) {
                    candidateSet.emplace(-dist, neighbor_node);
                    topCandidates.emplace(dist, neighbor_node);
                    if (topCandidates.size() > beam_size) topCandidates.pop();
                    if (!topCandidates.empty()) lowerBound = topCandidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        // return simply the neighbors
        std::vector<uint> neighbors(k);
        while (topCandidates.size() > k) topCandidates.pop();
        for (int i = k - 1; i >= 0; i--) {
            neighbors[i] = topCandidates.top().second;
            topCandidates.pop();
        }
        return neighbors;
    }

    // perform the hsp test to get the hsp neighbors of the node
    int edges_hsp_maxK_ = 0;
    std::vector<uint> edges_HSP(uint query, std::vector<uint> const& set) {
        std::vector<uint> neighbors{};
        char* queryPtr = getDataByIndex(query);

        // - initialize the active list A
        std::vector<std::pair<float, uint>> active_list{};
        active_list.reserve(set.size());

        // - initialize the list with all points and distances, find nearest neighbor
        uint index1;
        float distance_Q1 = HUGE_VAL;
        for (uint index : set) {
            if (index == query) continue;
            float const distance = compute_distance(queryPtr, getDataByIndex(index));
            if (distance < distance_Q1) {
                distance_Q1 = distance;
                index1 = index;
            }
            active_list.emplace_back(distance, index);
        }

        // - limit the number of points to consider
        if (edges_hsp_maxK_ > 0 && active_list.size() > edges_hsp_maxK_) {
            // - nth_element sort: bring the kth closest elements to the front, but not sorted, O(N)
            std::nth_element(active_list.begin(), active_list.begin() + edges_hsp_maxK_, active_list.end());
            active_list.resize(edges_hsp_maxK_);  // keep the top k points
        }

        // - perform the hsp loop
        while (active_list.size() > 0) {
            if (max_neighbors_ > 0 && neighbors.size() >= max_neighbors_) break;

            // - next neighbor as closest valid point
            neighbors.push_back(index1);
            char* index1_ptr = getDataByIndex(index1);

            // - set up for the next hsp neighbor
            uint index1_next;
            float distance_Q1_next = HUGE_VAL;

            // - initialize the active_list for next iteration
            // - make new list: push_back O(1) faster than deletion O(N)
            std::vector<std::pair<float, uint>> active_list_copy = active_list;
            active_list.clear();

            // - check each point for elimination
            for (int it2 = 0; it2 < (int)active_list_copy.size(); it2++) {
                uint const index2 = active_list_copy[it2].second;
                float const distance_Q2 = active_list_copy[it2].first;
                if (index2 == index1) continue;
                float const distance_12 = compute_distance(index1_ptr, getDataByIndex(index2));

                // - check the hsp inequalities: add if not satisfied
                if (distance_Q1 >= distance_Q2 || distance_12 >= distance_Q2) {
                    active_list.emplace_back(distance_Q2, index2);

                    // - update neighbor for next iteration
                    if (distance_Q2 < distance_Q1_next) {
                        distance_Q1_next = distance_Q2;
                        index1_next = index2;
                    }
                }
            }

            // - setup the next hsp neighbor
            index1 = index1_next;
            distance_Q1 = distance_Q1_next;
        }

        return neighbors;
    }

/*
|=======================================================================================================================
||
||
||                          SEARCHING
||
||
|=======================================================================================================================
*/

    // find knn by brute force
    std::priority_queue<std::pair<float, uint>> search_bruteForce(const void* queryPtr_v, int k = 1) {
        char* query_ptr = (char*) (queryPtr_v);
        if (dataset_size_ < k) std::runtime_error("Error: small dataset\n");


        // consider all points for priority queue
        std::priority_queue<std::pair<float, uint>> topResults;
        for (uint index = 0; index < (uint)k; index++) {
            float dist = compute_distance(query_ptr, index);
            topResults.emplace(dist, index);
        }
        float lastDist = topResults.top().first;
        for (uint index = k; index < dataset_size_; index++) {
            float dist = compute_distance(query_ptr, index);
            if (dist <= lastDist) {
                topResults.emplace(dist, index);
                if (topResults.size() > k) topResults.pop();
                lastDist = topResults.top().first;
            }
        }

        // collect the result
        while (topResults.size() > k) topResults.pop();
        return topResults;
    }

    // metrics
    double metric_time_hierarchy_{0};
    double metric_time_graph_{0};
    unsigned long int metric_distanceCount_{0};
    unsigned long int metric_numHops_{0};
    double metric_starting_distance_{0};

    // - perform search using graph
    std::priority_queue<std::pair<float, uint>> search(const void* query_ptr_v, int k = 1) {
        char* query_ptr = (char*) query_ptr_v;
        // if (num_levels_ == 0) std::runtime_error("No hierarchy initialized\n");
        
        // find entry-point in graph by top-down traversal of hierarchical partitioning
        // tStart = std::chrono::high_resolution_clock::now();
        uint start_node = 0;
        for (int c = 0; c < num_levels_; c++) {

            // - define the candidate pivots in the layer
            std::vector<uint> candidate_coarse_pivots{};
            if (c == 0) {
                candidate_coarse_pivots = hierarchy_[c].pivots_;
            } else {
                candidate_coarse_pivots = hierarchy_[c - 1].get_partition(start_node);
            }

            // - find and record the closest coarse pivot
            float closest_dist = HUGE_VAL;
            for (uint coarse_pivot : candidate_coarse_pivots) {
                float dist = compute_distance(query_ptr, coarse_pivot);
                // metric_distanceCount_++;
                if (dist < closest_dist) {
                    closest_dist = dist;
                    start_node = coarse_pivot;
                }
            }
        }
        // tEnd = std::chrono::high_resolution_clock::now();
        // metric_time_hierarchy_ += std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();

        // perform the beam search
        int beamSize = search_beam_size_;
        if (beamSize < k) beamSize = k;
        // tStart = std::chrono::high_resolution_clock::now();
        std::priority_queue<std::pair<float, uint>> topCandidates = beamSearch(query_ptr, start_node, beamSize);
        // tEnd = std::chrono::high_resolution_clock::now();
        // metric_time_graph_ += std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();

        while (topCandidates.size() > k) topCandidates.pop();
        return topCandidates;
    }

    // performing beam search on a graph on a specific layer
    std::priority_queue<std::pair<float, uint>> beamSearch(const void* query_ptr_v, uint start_node, int beam_size) {
        char* query_ptr = (char*) (query_ptr_v);
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // starting node
        float dist = compute_distance(query_ptr, start_node);
        // metric_starting_distance_ += dist;

        // intialize queues: topCandidates to hold beam, candidateSet to hold explorable nodes
        std::priority_queue<std::pair<float, uint>> topCandidates;  // top k points found (largest first)
        std::priority_queue<std::pair<float, uint>> candidateSet;   // points to explore (smallest first)
        topCandidates.emplace(dist, start_node);
        float lower_bound = dist;       // distance to kth neighbor
        candidateSet.emplace(-topCandidates.top().first, topCandidates.top().second);
        visited_array[start_node] = visited_array_tag;

        //> Perform the beam search loop
        while (!candidateSet.empty()) {
            uint candidate_node = candidateSet.top().second;
            float candidate_distance = -candidateSet.top().first;

            // if we have explored all points in our beam, stop
            // if (candidate_distance > lower_bound) break; // hnsw
            if ((candidate_distance > lower_bound) && (topCandidates.size() >= beam_size)) break;
            candidateSet.pop();
            // metric_numHops_++;

            // iterate through node neighbors
            uint num_neighbors;
            uint* neighbors = get_neighbors(candidate_node, num_neighbors);

            // prefetching:
            //      - bring in the tag from visiting array: uint short...
            //      - bring in the neighbor data: first neighbor
            //      - bring in the next neighbor
            // #ifdef USE_SSE
            //     _mm_prefetch((char *) (visited_array + *(neighbors)), _MM_HINT_T0);
            //     // _mm_prefetch((char *) (visited_array + *(neighbors) + 64), _MM_HINT_T0);
            //     _mm_prefetch(data_bottom_level_ + (*(neighbors))*size_per_element_ + size_links_, _MM_HINT_T0);
            //     _mm_prefetch((char *) (neighbors + 1), _MM_HINT_T0);
            // #endif

            for (int i = 0; i < num_neighbors; i++) {
                uint neighbor_node = neighbors[i];

                // prefetching:
                //      - bring in the next neighbor visiting array
                //      - bring in the next neighbor data
                // #ifdef USE_SSE
                //     _mm_prefetch((char *) (visited_array + *(neighbors + i)), _MM_HINT_T0);
                //     _mm_prefetch(data_bottom_level_ + (*(neighbors+i))*size_per_element_ + size_links_, _MM_HINT_T0);
                // #endif

                //  - break condition
                if (search_max_neighbors_ > 0 && i >= search_max_neighbors_) break;

                // - skip if already visisted
                if (visited_array[neighbor_node] == visited_array_tag) continue;
                visited_array[neighbor_node] = visited_array_tag;

                // add to beam if closer than some point in the beam, or beam not yet full
                float const dist = compute_distance(query_ptr, neighbor_node);
                // metric_distanceCount_++;
                if (topCandidates.size() < beam_size || dist <= lower_bound) {
                    candidateSet.emplace(-dist, neighbor_node);

                    // prefetching:
                    //      - bring in the next candidate node neighbors
                    // #ifdef USE_SSE
                    //     _mm_prefetch(data_bottom_level_ + candidateSet.top().second * size_per_element_, _MM_HINT_T0);
                    // #endif

                    // update beam
                    topCandidates.emplace(dist, neighbor_node);
                    if (topCandidates.size() > beam_size) topCandidates.pop();
                    lower_bound = topCandidates.top().first;
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return topCandidates;
    }
};