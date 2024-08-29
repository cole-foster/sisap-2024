#pragma once
#include <omp.h>

#include <algorithm>
#include <random>
#include <atomic>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>


#include "distances.h"
#include "visited_list_pool.h"
#include "utils.hpp"

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
    GraphHierarchy() {};
    GraphHierarchy(float*& data_pointer, uint dimension, uint dataset_size,
                   distances::SpaceInterface<float>* s) {
        dataPointer_ = data_pointer;
        dimension_ = (size_t)dimension;
        datasetSize_ = dataset_size;

        // using the distance function from hnswlib
        space_ = s;
        distFunc_ = s->get_dist_func();
        distFuncParam_ = s->get_dist_func_param();
        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, datasetSize_));

        srand(3);
    }
    GraphHierarchy(uint dimension, uint dataset_size, distances::SpaceInterface<float>* s) {
        dimension_ = (size_t) dimension;
        datasetSize_ = dataset_size;

        // need to own the data pointer
        ownDataPointer_ = true;
        dataPointer_ = new float[dimension_ * (size_t) datasetSize_];
        printf(" !!! Data not given: add each element with addPoint() command\n");

        // using the distance function from hnswlib
        space_ = s;
        distFunc_ = s->get_dist_func();
        distFuncParam_ = s->get_dist_func_param();
        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, datasetSize_));

        srand(3);
    }
    ~GraphHierarchy() {
        visited_list_pool_.reset(nullptr);
        if (ownDataPointer_) delete[] dataPointer_;
    };

    // dataset information
    bool ownDataPointer_ = false;
    float* dataPointer_{nullptr};
    size_t dimension_ = 0;  // size_t because N*dimension can be greater than limit of uint
    uint datasetSize_ = 0;

    // distances
    distances::SpaceInterface<float>* space_;
    DISTFUNC<float> distFunc_;
    void* distFuncParam_{nullptr};

    // stats
    std::chrono::high_resolution_clock::time_point tStart, tEnd;
    std::chrono::high_resolution_clock::time_point tStart1, tEnd1;
    std::chrono::high_resolution_clock::time_point tStart2, tEnd2;

    // distance computations
    float* getIndexDataPointer(uint index) const { return (dataPointer_ + (size_t)index * dimension_); }
    float compute_distance(float* index1_ptr, float* index2_ptr) const {
        return distFunc_(index1_ptr, index2_ptr, distFuncParam_);
    }
    float compute_distance(float* index1_ptr, uint index2) const {
        return distFunc_(index1_ptr, getIndexDataPointer(index2), distFuncParam_);
    }
    float compute_distance(uint index1, float* index2_ptr) const {
        return distFunc_(getIndexDataPointer(index1), index2_ptr, distFuncParam_);
    }
    float compute_distance(uint index1, uint index2) const {
        return distFunc_(getIndexDataPointer(index1), getIndexDataPointer(index2), distFuncParam_);
    }

    // add each data point and initialize the bottom layer graph
    void addPoint(float* data_point, uint element) {
        if (element >= datasetSize_) {
            throw std::runtime_error("Element ID exceeded datasetSize_");
        }
        // - initializing and setting data memory for bottom level
        memset(dataPointer_ + dimension_ * element, 0, dimension_ * sizeof(float));
        memcpy(dataPointer_ + dimension_ * element, data_point, dimension_ * sizeof(float));
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
    std::vector<std::vector<uint>> graph_{};
    int maxNeighbors_{0};

    void saveGraph(std::string filename) {
        if (graph_.size() == 0) return;
        printf("Saving graph to: %s\n", filename.c_str());
        Utils::saveGraph(filename, graph_, dimension_, datasetSize_);
        return;
    }
    void clearGraph() {
        graph_.clear();
        printGraphStats();
    }
    void loadGraph(std::string filename) {
        printf("Loading graph from: %s\n", filename.c_str());
        Utils::loadGraph(filename, graph_, dimension_, datasetSize_);
        printGraphStats();
    }
    void printGraphStats() {
        printf("Graph Statistics:\n");
        if (graph_.size() <= 0) {
            printf(" * No graph initialized...\n");
            return;
        }
        printf(" * number of nodes: %u\n", graph_.size());

        // - get the average degree
        size_t min_degree = 1000000;
        size_t max_degree = 0;
        size_t num_edges = 0.0f;
        for (uint node = 0; node < graph_.size(); node++) {
            size_t node_edges = graph_[node].size();
            num_edges += node_edges;
            if (node_edges < min_degree) min_degree = node_edges;
            if (node_edges > max_degree) max_degree = node_edges;
        }
        printf(" * number of edges: %lu\n", num_edges);
        double ave_degree = (double)num_edges / (double)datasetSize_;
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
    void constructExactHSPGraph(uint maxNeighborhoodSize = 0, int maxNeighbors = 0) {
        maxNeighbors_ = maxNeighbors;
        printf("Begin HSP Graph Construction\n");
        printf(" * N = %u\n", datasetSize_);
        if (maxNeighbors_ == 0) {
            printf(" * maxNeighbors: infinity\n");
        } else {
            printf(" * maxNeighbors: %u\n", maxNeighbors_);
        }
        if (maxNeighborhoodSize == 0) {
            printf(" * Exact HSP!\n");
        } else {
            printf(" * kNN-Constrained HSP with k=%u\n", maxNeighborhoodSize);
        }
        fflush(stdout);
        graph_.clear();
        graph_.resize(datasetSize_);

        std::vector<uint> datasetVector(datasetSize_);
        std::iota(datasetVector.begin(), datasetVector.end(), 0);

        // Apply HSP test to each node in parallel
        tStart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (uint node = 0; node < datasetSize_; node++) {
            graph_[node] = HSPTest(node, datasetVector, maxNeighborhoodSize);
        }
        tEnd = std::chrono::high_resolution_clock::now();
        double time_graph = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
        printf(" * Graph Construction Time (s): %.4f\n", time_graph);
        fflush(stdout);
        printGraphStats();
    }


    // perform the hsp test to get the hsp neighbors of the node
    std::vector<unsigned int> HSPTest(uint const query, std::vector<uint> const& set, int maxK = 0) const {
        std::vector<unsigned int> neighbors{};
        float* queryPtr = getIndexDataPointer(query);

        // - initialize the active list A
        std::vector<std::pair<float, uint>> active_list{};
        active_list.reserve(set.size());

        // - initialize the list with all points and distances, find nearest neighbor
        uint index1;
        float distance_Q1 = HUGE_VAL;
        for (uint index : set) {
            if (index == query) continue;
            float const distance = compute_distance(queryPtr, getIndexDataPointer(index));
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
            if (maxNeighbors_ > 0 && neighbors.size() >= maxNeighbors_) break;

            // - next neighbor as closest valid point
            neighbors.push_back(index1);
            float* index1_ptr = getIndexDataPointer(index1);

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
                float const distance_12 = compute_distance(index1_ptr, getIndexDataPointer(index2));

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
    std::vector<Level> hierarchy_{};
    int num_levels_{0};
    std::default_random_engine level_generator_;

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    // initialize the hierarchy
    int random_seed_ = 100;
    void initializeHierarchy(int const scaling, int num_levels=0) {
        hierarchy_.clear();

        // estimate the number of levels
        num_levels_ = num_levels;
        if (num_levels_ <= 0) {
            double start_value = (double) datasetSize_;
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
        for (uint index = 0; index < datasetSize_; index++) {
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
        printf("    - %d --> %u (bottom)\n", num_levels_ - 1, datasetSize_);

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
            printf("    - num_pivots = %u\n", (uint) num_pivots);
            fflush(stdout);
            hierarchy_[ell - 1].initialize_partitions();
            std::vector<uint> pivot_assignments(num_pivots);

            //> Perform the partitioning of this current layer
            // - perform this task using all threads
            #pragma omp parallel for
            for (size_t itp = 0; itp < num_pivots; itp++) {
                const uint fine_pivot = hierarchy_[ell].pivots_[itp];
                float* fine_pivot_ptr = getIndexDataPointer(fine_pivot);
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
            printf("    - num elements = %u\n", (uint) datasetSize_);
            fflush(stdout);
            tStart1 = std::chrono::high_resolution_clock::now();

            // - initializations
            hierarchy_[ell - 1].initialize_partitions();
            std::vector<uint> pivot_assignments(datasetSize_);
            
            #pragma omp parallel for
            for (uint node = 0; node < datasetSize_; node++) {
                float* node_ptr = getIndexDataPointer(node);
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
            for (uint node = 0; node < datasetSize_; node++) {
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

        // // Verify coverage
        // {
        //     printf(" * checking hierarchy coverage...\n");
        //     bool flag_coverage = true;
        //     std::vector<int> coverage(datasetSize_, 0);
        //     for (uint pivot : hierarchy_[num_levels_ - 2].pivots_) {
        //         std::vector<uint> const& partish = hierarchy_[num_levels_ - 2].get_partition(pivot);
        //         for (uint member : partish) {
        //             coverage[member] = 1;
        //         }
        //     }
        //     // check
        //     for (size_t i = 0; i < datasetSize_; i++) {
        //         if (coverage[i] == 0) {
        //             printf("No coverage of index: %u\n", i); 
        //             flag_coverage = false;
        //             break;
        //         }
        //     }
        //     printf(" * coverage: %u\n", flag_coverage);
        // }

        return;
    }

    // Hierarchical partitioning for search time
    void constructPartitioning2L(int const scaling) {
        printf("|> Creating 2-Layer hierarchical partitioning...\n");
        fflush(stdout);
        tStart1 = std::chrono::high_resolution_clock::now();

        // initilaize the hierarchy, 2 layer
        initializeHierarchy(scaling, 2);

        //> Assign each point to their closest observer
        hierarchy_[0].initialize_partitions();
        size_t num_pivots = hierarchy_[0].num_pivots_;
        printf(" * Partitioning %u points to %u pivots\n", datasetSize_, num_pivots);
        fflush(stdout);

        //> Find the closest pivot to each node
        std::vector<uint> pivot_assignments(datasetSize_);
#pragma omp parallel for
        for (uint node = 0; node < datasetSize_; node++) {
            uint closest_pivot;
            float closest_dist = HUGE_VAL;

            for (uint pivot : hierarchy_[0].pivots_) {
                float const dist = compute_distance(node, pivot);
                if (dist < closest_dist) {
                    closest_dist = dist;
                    closest_pivot = pivot;
                }
            }

            // - record the closest coarse pivot found (thread safe)
            pivot_assignments[node] = closest_pivot;
        }

        //> Assign each node for the partitioning!
        for (uint node = 0; node < datasetSize_; node++) {
            uint parent = pivot_assignments[node];
            hierarchy_[0].add_member(parent, node);
        }

        tEnd1 = std::chrono::high_resolution_clock::now();
        double time_partition = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd1 - tStart1).count();
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
    void buildGraph(int scaling = 100, int b = 40, std::string edgeType = "hsp") {
        printf("Begin Index Construction...\n");
        tStart = std::chrono::high_resolution_clock::now();

        printf(" * edge selection: %s\n", edgeType.c_str());
        std::vector<uint> (GraphHierarchy::*edge_func)(uint, std::vector<uint> const&); 
        if (edgeType == "complete") {
            edge_func = &GraphHierarchy::edges_complete;
        } else if (edgeType == "hsp") {
            edge_func = &GraphHierarchy::edges_HSP;
        } else if (edgeType == "knn") {
            printf(" * knn: please set member variable `edges_knn_k_`=%d\n", edges_knn_k_);
            edge_func = &GraphHierarchy::edges_knn;
        } else {
            printf(" * Unknown type! Defaulting to hsp\n");
            edge_func = &GraphHierarchy::edges_HSP;
        }

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
            printf("    - num_pivots = %u\n", (uint) num_pivots);
            fflush(stdout);
            hierarchy_[ell - 1].initialize_partitions();

            // graph needed
            bool flag_graph_needed = false;
            if (num_pivots * scaling >= 20000) flag_graph_needed = true;
            bool flag_coarse_graph = false;
            if (hierarchy_[ell-1].graph_.size() > 0) flag_coarse_graph = true;

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
                    float* pivot_ptr = getIndexDataPointer(pivot);

                    // find closest partition
                    uint closest_pivot;
                    float closest_dist = HUGE_VAL;
                    for (uint coarse_pivot : hierarchy_[ell-1].pivots_) {
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
                float* pivot_ptr = getIndexDataPointer(pivot);
                uint closest_coarse_pivot; // to find, for assignment

                // - define the candidate list of neighbors
                if (!flag_coarse_graph) { // brute force
                    float closest_coarse_dist = HUGE_VAL;
                    for (uint coarse_pivot : hierarchy_[ell-1].pivots_) {
                        float dist = compute_distance(pivot_ptr, coarse_pivot);
                        if (dist < closest_coarse_dist) {
                            closest_coarse_dist = dist;
                            closest_coarse_pivot = coarse_pivot;
                        }
                    }
                    vec_closest_coarse_pivots[itp] = {closest_coarse_pivot};
                } else { // by graph

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
                    vec_closest_coarse_pivots[itp] = beamSearchConstruction(ell - 1, pivot_ptr, b, closest_coarse_pivot);
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
                        std::vector<uint> const &coarse_partition = hierarchy_[ell - 1].get_partition(coarse_pivot);
                        candidate_region.insert(candidate_region.end(), coarse_partition.begin(), coarse_partition.end());
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
            printf("    - num elements = %u\n", (uint) datasetSize_);
            fflush(stdout);

            // graph needed
            printf("    - creating graph and partitioning in tandem\n");
            bool flag_coarse_graph = false;
            if (hierarchy_[ell-1].graph_.size() > 0) flag_coarse_graph = true;
            hierarchy_[ell - 1].initialize_partitions();
            graph_.resize(datasetSize_);

            //> Find the closest observers
            tStart2 = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<uint>> vec_closest_coarse_pivots(datasetSize_);
            #pragma omp parallel for
            for (uint node = 0; node < datasetSize_; node++) {
                float* node_ptr = getIndexDataPointer(node);

                // - define the candidate list of neighbors
                if (!flag_coarse_graph) { // brute force

                    uint closest_coarse_pivot; 
                    float closest_coarse_dist = HUGE_VAL;
                    for (uint coarse_pivot : hierarchy_[ell-1].pivots_) {
                        float dist = compute_distance(node_ptr, coarse_pivot);
                        if (dist < closest_coarse_dist) {
                            closest_coarse_dist = dist;
                            closest_coarse_pivot = coarse_pivot;
                        }
                    }
                    vec_closest_coarse_pivots[node] = {closest_coarse_pivot};
                } else { // by graph

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
                    vec_closest_coarse_pivots[node] = beamSearchConstruction(ell - 1, node_ptr, b, closest_coarse_pivot);
                }
            }
            // - assign to the partitions (thread safe)
            for (uint node = 0; node < datasetSize_; node++) {
                hierarchy_[ell - 1].add_member(vec_closest_coarse_pivots[node][0], node);
            }
            tEnd2 = std::chrono::high_resolution_clock::now();
            double time_partition = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd2 - tStart2).count();
            printf("    - level-%d partition time (s): %.4f\n", ell, time_partition);
            fflush(stdout);

            // - now perform the graph construction
            tStart2 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (uint node = 0; node < datasetSize_; node++) {
                float* node_ptr = getIndexDataPointer(node);

                // - define the candidate region
                std::vector<uint> candidate_region{};
                if (!flag_coarse_graph) {
                    candidate_region = hierarchy_[ell].pivots_;
                } else {
                    for (uint coarse_pivot : vec_closest_coarse_pivots[node]) {
                        std::vector<uint> const &coarse_partition = hierarchy_[ell - 1].get_partition(coarse_pivot);
                        candidate_region.insert(candidate_region.end(), coarse_partition.begin(), coarse_partition.end());
                    }
                }

                // - perform the edge selection test on this neighborhood
                std::vector<uint> neighbors = (this->*edge_func)(node, candidate_region);
                graph_[node] = neighbors;
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
    int beamSizeConstruction_ = 10;
    void set_beamSizeConstruction(int beamSizeConstruction) {beamSizeConstruction_ = beamSizeConstruction;}
    
    std::vector<uint> beamSearchConstruction(int level, float* query_ptr, int k, uint start_node) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
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
            std::vector<uint> const &current_node_neighbors = hierarchy_[level].get_neighbors(current_node);
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

    // graph parameters
    void buildGraph_2L(int scaling = 100, int b = 40, std::string edgeType = "hsp") {
        printf("Begin Index Construction...\n");

        printf(" * edge selection: %s\n", edgeType.c_str());
        std::vector<uint> (GraphHierarchy::*edge_func)(uint, std::vector<uint> const&); 
        if (edgeType == "complete") {
            edge_func = &GraphHierarchy::edges_complete;
        } else if (edgeType == "hsp") {
            edge_func = &GraphHierarchy::edges_HSP;
        } else if (edgeType == "rng") {
            edge_func = &GraphHierarchy::edges_RNG;
        } else if (edgeType == "knn") {
            printf(" * knn: please set member variable `edges_knn_k_`=%d\n", edges_knn_k_);
            edge_func = &GraphHierarchy::edges_knn;
        } else if (edgeType == "range") {
            printf(" * range: please set member variable `edges_range_r_`=%.4f\n", edges_range_r_);
            edge_func = &GraphHierarchy::edges_range;
        } else if (edgeType == "hyperbolic") {
            printf(" * hyperbolic: please set member variable `edges_hyperbolic_epsilon_`=%.4f\n", edges_hyperbolic_epsilon_);
            edge_func = &GraphHierarchy::edges_hyperbolic;
        } else if (edgeType == "hilbert") {
            printf(" * hilbert: please set member variable `edges_hilbert_epsilon_`=%.4f\n", edges_hilbert_epsilon_);
            edge_func = &GraphHierarchy::edges_hilbert;
        } else if (edgeType == "vanama") {
            printf(" * vanama: please set member variable `edges_vanama_alpha_`=%.4f\n", edges_vanama_alpha_);
            edge_func = &GraphHierarchy::edges_vanama;
        } else if (edgeType == "random") {
            printf(" * random: please set member variable `edges_random_k_`=%.4f\n", edges_random_k_);
            edge_func = &GraphHierarchy::edges_random;
        } else {
            printf(" * Unknown type! Defaulting to hsp\n");
            edge_func = &GraphHierarchy::edges_HSP;
        }
        tStart = std::chrono::high_resolution_clock::now();

        //> Initialize Hierarchy and Perform Partitioning
        hierarchy_.clear();
        constructPartitioning2L(scaling);

        printf("|> Creating bottom-layer graph...\n");
        tStart1 = std::chrono::high_resolution_clock::now();
        graph_.clear();
        graph_.resize(datasetSize_);

        //> Find Approximate HSP Neighbors of Each Node in Parallel
        #pragma omp parallel for
        for (uint node = 0; node < datasetSize_; node++) {

            // identify the b closest pivots
            std::priority_queue<std::pair<float, uint>> closestPivots;
            for (uint pivot : hierarchy_[0].pivots_) {
                float const dist = compute_distance(node, pivot);
                closestPivots.emplace(dist, pivot);
                if (closestPivots.size() > b) closestPivots.pop();
            }

            // - collect partitions associated with the closest coarse pivots
            std::vector<uint> candidate_nodes{};
            while (closestPivots.size() > 0) {
                uint pivot = closestPivots.top().second;
                closestPivots.pop();
                std::vector<uint> const& pivot_partition = hierarchy_[0].get_partition(pivot);
                candidate_nodes.insert(candidate_nodes.end(), pivot_partition.begin(), pivot_partition.end());
            }

            // - Perform the HSP test on this neighborhood
            graph_[node] = (this->*edge_func)(node, candidate_nodes);
        }
        tEnd = std::chrono::high_resolution_clock::now();
        double time_graph = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart1).count();
        printf(" * Graph construction time (s): %.4f\n", time_graph);
        double time_total = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
        printf(" * Total index construction time (s): %.4f\n", time_total);
        fflush(stdout);
        printGraphStats();
        return;
    }

/*
|=======================================================================================================================
||
||
||                          EDGE SELECTION FUNCTIONS
||
||
|=======================================================================================================================
*/

    std::vector<uint> edges_complete(uint query, std::vector<uint> const& set) {
        return set;
    }


    // perform the hsp test to get the hsp neighbors of the node
    int edges_hsp_maxK_ = 0;
    std::vector<uint> edges_HSP(uint query, std::vector<uint> const& set) {
        std::vector<uint> neighbors{};
        float* queryPtr = getIndexDataPointer(query);

        // - initialize the active list A
        std::vector<std::pair<float, uint>> active_list{};
        active_list.reserve(set.size());

        // - initialize the list with all points and distances, find nearest neighbor
        uint index1;
        float distance_Q1 = HUGE_VAL;
        for (uint index : set) {
            if (index == query) continue;
            float const distance = compute_distance(queryPtr, getIndexDataPointer(index));
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
            if (maxNeighbors_ > 0 && neighbors.size() >= maxNeighbors_) break;

            // - next neighbor as closest valid point
            neighbors.push_back(index1);
            float* index1_ptr = getIndexDataPointer(index1);

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
                float const distance_12 = compute_distance(index1_ptr, getIndexDataPointer(index2));

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

    // perform the hsp test to get the hsp neighbors of the node
    std::vector<uint> edges_RNG(uint query, std::vector<uint> const& set) {
        std::vector<uint> neighbors{};
        float* queryPtr = getIndexDataPointer(query);

        // - get the hsp neighbors first
        std::vector<uint> hsp_neighbors = edges_HSP(query, set);

        // - only keep hsp neighbors with no interference
        for (uint index2 : hsp_neighbors) {
            bool flag_keep = true;
            float distQ2 = compute_distance(queryPtr, index2);

            // check all for intereference
            for (uint index3 : set) {
                if (index3 == query || index3 == index2) continue;
                float distQ3 = compute_distance(queryPtr, index3);
                float dist23 = compute_distance(index2, index3);

                if (distQ3 < distQ2 && dist23 < distQ2) {
                    flag_keep = false;
                    break;
                }
            }
            if (flag_keep) {
                neighbors.push_back(index2);
            }
        }

        return neighbors;
    }

    // perform the hsp test to get the hsp neighbors of the node
    int edges_knn_k_ = 10;
    std::vector<uint> edges_knn(uint query, std::vector<uint> const& set) {
        std::vector<uint> neighbors;
        int k = edges_knn_k_;
        if (set.size() <= k) return set;
        float* queryPtr = getIndexDataPointer(query);

        // add and update this priority queue
        std::priority_queue<std::pair<float, uint>> topResults;
        for (size_t it = 0; it < set.size(); it++) {
            uint const index = set[it];
            if (index == query) continue;
            float dist = compute_distance(queryPtr, index);
            if (topResults.size() < k || dist <= topResults.top().first) {
                topResults.emplace(dist, index);
                if (topResults.size() > k) topResults.pop();
            }
        }

        // collect the result
        neighbors.resize(k);
        for (int i = k - 1; i >= 0; i--) {
            neighbors[i] = topResults.top().second;
            topResults.pop();
        }
        return neighbors;
    }

       // perform the hsp test to get the hsp neighbors of the node
    float edges_range_r_ = 0;
    std::vector<uint> edges_range(uint query, std::vector<uint> const& set) {
        std::vector<uint> neighbors;
        float r = edges_range_r_;
        float* queryPtr = getIndexDataPointer(query);

        // add and update this priority queue
        for (size_t it = 0; it < set.size(); it++) {
            uint const index = set[it];
            if (index == query) continue;
            float dist = compute_distance(queryPtr, index);
            if (dist <= r) {
                neighbors.push_back(index);
            }
        }

        return neighbors;
    }

    // perform the hsp test to get the hsp neighbors of the node
    float edges_vanama_alpha_ = 1.0f;
    std::vector<uint> edges_vanama(uint query, std::vector<uint> const& set) {
        std::vector<uint> neighbors{};
        float* queryPtr = getIndexDataPointer(query);

        // - initialize the active list A
        std::vector<std::pair<float, uint>> active_list{};
        active_list.reserve(set.size());

        // - initialize the list with all points and distances, find nearest neighbor
        uint index1;
        float distance_Q1 = HUGE_VAL;
        for (uint index : set) {
            if (index == query) continue;
            float const distance = compute_distance(queryPtr, getIndexDataPointer(index));
            if (distance < distance_Q1) {
                distance_Q1 = distance;
                index1 = index;
            }
            active_list.emplace_back(distance, index);
        }

        // - perform the hsp loop
        while (active_list.size() > 0) {
            if (maxNeighbors_ > 0 && neighbors.size() >= maxNeighbors_) break;

            // - next neighbor as closest valid point
            neighbors.push_back(index1);
            float* index1_ptr = getIndexDataPointer(index1);

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
                float const distance_12 = compute_distance(index1_ptr, getIndexDataPointer(index2));

                // - check the hsp inequalities: add if not satisfied
                if (edges_vanama_alpha_ * distance_12 >= distance_Q2) { // Vanama inequality
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

    // perform the hsp test to get the hsp neighbors of the node
    float edges_hyperbolic_epsilon_ = 0.0f;
    std::vector<uint> edges_hyperbolic(uint query, std::vector<uint> const& set) {
        std::vector<uint> neighbors{};
        float* queryPtr = getIndexDataPointer(query);

        // - initialize the active list A
        std::vector<std::pair<float, uint>> active_list{};
        active_list.reserve(set.size());

        // - initialize the list with all points and distances, find nearest neighbor
        uint index1;
        float distance_Q1 = HUGE_VAL;
        for (uint index : set) {
            if (index == query) continue;
            float const distance = compute_distance(queryPtr, getIndexDataPointer(index));
            if (distance < distance_Q1) {
                distance_Q1 = distance;
                index1 = index;
            }
            active_list.emplace_back(distance, index);
        }

        // - perform the hsp loop
        while (active_list.size() > 0) {
            if (maxNeighbors_ > 0 && neighbors.size() >= maxNeighbors_) break;

            // - next neighbor as closest valid point
            neighbors.push_back(index1);
            float* index1_ptr = getIndexDataPointer(index1);

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
                float const distance_12 = compute_distance(index1_ptr, getIndexDataPointer(index2));

                // - super monotonic: squared distances
                if (distance_Q2 - distance_12 <= 2*edges_hyperbolic_epsilon_) { // hyperbolic
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

    // perform the hsp test to get the hsp neighbors of the node
    float edges_hilbert_epsilon_ = 0.0f;
    std::vector<uint> edges_hilbert(uint query, std::vector<uint> const& set) {
        std::vector<uint> neighbors{};
        float* queryPtr = getIndexDataPointer(query);

        // - initialize the active list A
        std::vector<std::pair<float, uint>> active_list{};
        active_list.reserve(set.size());

        // - initialize the list with all points and distances, find nearest neighbor
        uint index1;
        float distance_Q1 = HUGE_VAL;
        for (uint index : set) {
            if (index == query) continue;
            float const distance = compute_distance(queryPtr, getIndexDataPointer(index));
            if (distance < distance_Q1) {
                distance_Q1 = distance;
                index1 = index;
            }
            active_list.emplace_back(distance, index);
        }

        // - perform the hsp loop
        while (active_list.size() > 0) {
            if (maxNeighbors_ > 0 && neighbors.size() >= maxNeighbors_) break;

            // - next neighbor as closest valid point
            neighbors.push_back(index1);
            float* index1_ptr = getIndexDataPointer(index1);

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
                float const distance_12 = compute_distance(index1_ptr, getIndexDataPointer(index2));

                // - check the hsp inequalities: add if not satisfied
                if (distance_Q2*distance_Q2 - distance_12*distance_12 <= 2*edges_hilbert_epsilon_*distance_Q1) { // hilbert
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

    // perform the hsp test to get the hsp neighbors of the node
    int edges_random_k_ = 10;
    std::vector<uint> edges_random(uint query, std::vector<uint> const& set) {
        std::vector<uint> neighbors{};
        
        std::mt19937 gen(query); // Seed the generator
        std::uniform_int_distribution<> distr(0, set.size()-1);
        for (int i = 0; i < edges_random_k_; /* iterate in loop*/) {
            int rand_idx = distr(gen);
            uint neighbor = set[rand_idx];
            if (std::find(neighbors.begin(), neighbors.end(), neighbor) == neighbors.end()) {
                neighbors.push_back(neighbor);
                i++;
            }
        }

        return neighbors;
    }




/*
|=======================================================================================================================
||
||
||                          SEARCH ON GRAPH
||
||
|=======================================================================================================================
*/
    // parameters
    int beamSize_{1};
    int maxNeighborsSearch_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // metrics
    unsigned long int metric_distanceCount_{0};
    unsigned long int metric_numHops_{0};
    double metric_starting_distance_{0};
    double metric_time_graph_{0};

    // performing beam search on a graph on a specific layer
    std::priority_queue<std::pair<float, uint>> beamSearch(float* query_ptr, uint start_node, int beam_size) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // intialize queues: topCandidates to hold beam, candidateSet to hold explorable nodes
        std::priority_queue<std::pair<float, uint>> topCandidates;  // top k points found (largest first)
        std::priority_queue<std::pair<float, uint>> candidateSet;  // points to explore (smallest first)
        float dist = compute_distance(query_ptr, start_node);
        topCandidates.emplace(dist, start_node);
        float lower_bound = dist;  // distance to kth neighbor
        candidateSet.emplace(-topCandidates.top().first, topCandidates.top().second);
        visited_array[start_node] = visited_array_tag;

        //> Perform the beam search loop
        while (!candidateSet.empty()) {
            uint const candidate_node = candidateSet.top().second;
            float const candidate_distance = -candidateSet.top().first;

            // if we have explored all points in our beam, stop
            // ensuring beam is full: not necessary according to hnsw
            if ((candidate_distance > lower_bound) && (topCandidates.size() >= beam_size))
                break;
            // if (candidate_distance > lower_bound)
                // break;
            candidateSet.pop();
            // metric_numHops_++;

            // iterate through node neighbors
            std::vector<uint> const& candidate_node_neighbors = graph_[candidate_node];
            size_t num_neighbors = candidate_node_neighbors.size();
            for (int i = 0; i < num_neighbors; i++) {
                uint neighbor_node = candidate_node_neighbors[i];

                // - skip if already visisted
                if (visited_array[neighbor_node] == visited_array_tag) continue;
                visited_array[neighbor_node] = visited_array_tag;
                if (maxNeighborsSearch_ > 0 && i >= maxNeighborsSearch_) continue;

                // add to beam if closer than some point in the beam, or beam not yet full
                float const dist = compute_distance(query_ptr, neighbor_node);
                // metric_distanceCount_++;
                if (topCandidates.size() < beam_size || dist <= lower_bound) {

                    // update notes to explore
                    candidateSet.emplace(-dist, neighbor_node);

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

    std::vector<uint> search_bruteForce(float* queryPtr, int k = 1) {
        std::vector<uint> neighbors(k);
        if (datasetSize_ == 0) return neighbors;

        std::priority_queue<std::pair<float, uint>> topResults;
        for (uint index = 0; index < (uint)k; index++) {
            float dist = compute_distance(queryPtr, index);
            topResults.emplace(dist, index);
        }
        float lastDist = topResults.top().first;
        for (uint index = k; index < datasetSize_; index++) {
            float dist = compute_distance(queryPtr, index);
            if (dist <= lastDist) {
                topResults.emplace(dist, index);
                if (topResults.size() > k) topResults.pop();
                lastDist = topResults.top().first;
            }
        }

        // collect the result
        while (topResults.size() > k) topResults.pop();
        for (int i = k - 1; i >= 0; i--) {
            neighbors[i] = topResults.top().second;
            topResults.pop();
        }
        return neighbors;
    }

    std::vector<uint> search_startClosestPivot(float* query_ptr, int k = 1) {
        if (num_levels_ == 0) {
            std::runtime_error("No hierarchy initialized\n");
        }

        // select a start node
        uint start_node = 0;
        float start_dist = HUGE_VAL;
        if (num_levels_ >= 2) {
            for (uint pivot : hierarchy_[num_levels_ - 2].pivots_) {
                float dist = compute_distance(query_ptr, pivot);
                if (dist < start_dist) {
                    start_dist = dist;
                    start_node = pivot;
                }
            }
        }

        // perform the beam search
        int beamSize = beamSize_;
        if (beamSize < k) beamSize = k;
        tStart = std::chrono::high_resolution_clock::now();  // timing the search
        std::priority_queue<std::pair<float, uint>> topCandidates = beamSearch(query_ptr, start_node, beamSize);
        tEnd = std::chrono::high_resolution_clock::now();
        metric_time_graph_ += std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();

        // collect the result
        std::vector<uint> neighbors(k);
        while (topCandidates.size() > k) topCandidates.pop();
        for (int i = k - 1; i >= 0; i--) {
            neighbors[i] = topCandidates.top().second;
            topCandidates.pop();
        }
        return neighbors;
    }

    double time_search_hierarchy_ = 0.0f;
    double time_search_graph_ = 0.0f;
    std::priority_queue<std::pair<float, uint>> search(float* query_ptr, int k = 1) {
        if (num_levels_ == 0) {
            std::runtime_error("No hierarchy initialized\n");
        }

        // top-down navigation to approx. closest node
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
                if (dist < closest_dist) {
                    closest_dist = dist;
                    start_node = coarse_pivot;
                }
            }
        }
        // tEnd = std::chrono::high_resolution_clock::now();
        // time_search_hierarchy_ += std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();

        // perform the beam search
        int beamSize = beamSize_;
        if (beamSize < k) beamSize = k;
        // tStart = std::chrono::high_resolution_clock::now();
        std::priority_queue<std::pair<float, uint>> topCandidates = beamSearch(query_ptr, start_node, beamSize);
        // tEnd = std::chrono::high_resolution_clock::now();
        // time_search_graph_ += std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();

        // collect the result
        // std::vector<uint> neighbors(k);
        while (topCandidates.size() > k) topCandidates.pop();
        // for (int i = k - 1; i >= 0; i--) {
        //     neighbors[i] = topCandidates.top().second;
        //     topCandidates.pop();
        // }
        return topCandidates;
    }


};