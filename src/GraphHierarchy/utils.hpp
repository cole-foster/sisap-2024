#pragma once

#include <fstream>
#include <vector>
#include <sys/stat.h>

namespace Utils {

inline bool fileExists(std::string filename) {
    struct stat buffer;   
    return (stat (filename.c_str(), &buffer) == 0); 
}

// Save a graph to a binary file. Saves as format
//      D, N, 
//      num_neighbors1, neighbor_1, neighbor_2, .... , neighbor_n
//      num_neighbors2, neighbor_1, neighbor_2, .... , neighbor_n
//      ...
//      num_neighborsN, neighbor_1, neighbor_2, .... , neighbor_n
//       
void saveGraph(std::string filename, std::vector<std::vector<unsigned int>> const& graph, unsigned int const dimension, unsigned int const dataset_size) {
    printf("Saving Graph to: %s\n", filename.c_str());
    std::ofstream outputFileStream(filename, std::ios::binary);
    if (!outputFileStream.is_open()) {
        printf("Open File Error\n");
        exit(0);
    }
    if ((unsigned int) graph.size() != dataset_size) {
        printf("Graph Size Does Not Match Given Dataset Size\n");
        exit(0);
    }

    // output the dimension and dataset size
    outputFileStream.write((char*)&dimension, sizeof(unsigned int));
    outputFileStream.write((char*)&dataset_size, sizeof(unsigned int));

    // save the neighbors of each element in the dataset
    //      num_neighbors, n1, n2, ..., nn
    for (unsigned int index = 0; index < dataset_size; index++) {
        unsigned int const num_neighbors = (unsigned int)graph[index].size();
        outputFileStream.write((char*)&num_neighbors, sizeof(unsigned int));

        // change to use vector.data()
        for (unsigned int j = 0; j < num_neighbors; j++) {
            unsigned int const neighbor_id = graph[index][j];
            outputFileStream.write((char*)&neighbor_id, sizeof(unsigned int));
        }
    }

    // done!
    outputFileStream.close();
    return;
}

// Load a graph from a binary file. All uints. Must be in format
//      D, N, 
//      num_neighbors1, neighbor_1, neighbor_2, .... , neighbor_n
//      num_neighbors2, neighbor_1, neighbor_2, .... , neighbor_n
//      ...
//      num_neighborsN, neighbor_1, neighbor_2, .... , neighbor_n
//       
void loadGraph(std::string filename, std::vector<std::vector<unsigned int>>& graph, unsigned int const dimension, unsigned int const dataset_size) {
    printf("Loading Graph from: %s\n", filename.c_str());
    std::ifstream inputFileStream(filename, std::ios::binary);
    if (!inputFileStream.is_open()) {
        printf("Open File Error\n");
        exit(0);
    }

    // first, read the dimension and dataset sizeof the dataset
    unsigned int graph_dimension;
    unsigned int graph_dataset_size;
    inputFileStream.read((char*)&graph_dimension, sizeof(unsigned int));
    inputFileStream.read((char*)&graph_dataset_size, sizeof(unsigned int));
    if (graph_dimension != dimension) {
        printf("Graph dimension not consistent with dataset dimension!\n");
        exit(0);
    }
    if (graph_dataset_size != dataset_size) {
        printf("Graph dataset size not consistent with dataset dataset size!\n");
        exit(0);
    }

    // initialize the graph for all members of the dataset
    graph.resize(dataset_size);

    // get all neighbors
    for (unsigned int index = 0; index < dataset_size; index++) {
        unsigned int num_neighbors;
        inputFileStream.read((char*)&num_neighbors, sizeof(unsigned int));
        graph[index].resize(num_neighbors);
        for (unsigned int j = 0; j < num_neighbors; j++) {
            inputFileStream.read((char*)&graph[index][j], sizeof(unsigned int));
        }
    }

    // done!
    inputFileStream.close();
    return;
}


};