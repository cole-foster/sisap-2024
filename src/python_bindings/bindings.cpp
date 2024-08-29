#include <assert.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>

#include <atomic>
#include <iostream>
#include <thread>

#include "graph-hierarchy.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.", buffer.ndim);
        throw std::runtime_error(msg);
    }
    if (buffer.ndim == 2) {
        *rows = buffer.shape[0];
        *features = buffer.shape[1];
    } else {
        *rows = 1;
        *features = buffer.shape[0];
    }
}

inline std::vector<size_t> get_input_ids_and_check_shapes(const py::object& ids_, size_t feature_rows) {
    std::vector<size_t> ids;
    if (!ids_.is_none()) {
        py::array_t<size_t, py::array::c_style | py::array::forcecast> items(ids_);
        auto ids_numpy = items.request();
        // check shapes
        if (!((ids_numpy.ndim == 1 && ids_numpy.shape[0] == feature_rows) ||
              (ids_numpy.ndim == 0 && feature_rows == 1))) {
            char msg[256];
            snprintf(msg, sizeof(msg), "The input label shape %d does not match the input data vector shape %d",
                     ids_numpy.ndim, feature_rows);
            throw std::runtime_error(msg);
        }
        // extract data
        if (ids_numpy.ndim == 1) {
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for (size_t i = 0; i < ids1.size(); i++) {
                ids1[i] = items.data()[i];
            }
            ids.swap(ids1);
        } else if (ids_numpy.ndim == 0) {
            ids.push_back(*items.data());
        }
    }

    return ids;
}

class Index {
   public:
    // multithreading
    int num_threads_default;

    // space
    distances::SpaceInterface<float>* space;
    std::string space_name;
    int dimension;
    bool normalize = false;

    // index
    GraphHierarchy* alg_;
    uint cur_l;

    // search
    size_t default_ef;

    Index(const std::string& space_name, const int dimension) : space_name(space_name), dimension(dimension) {
        normalize = false;
        if (space_name == "l2") {
            space = new distances::L2Space(dimension);
        } else if (space_name == "ip") {
            space = new distances::InnerProductSpace(dimension);
        } 
        // else if (space_name == "cosine") {
        //     space = new distances::InnerProductSpace(dimension);
        //     normalize = true;
        // } 
        else {
            throw std::runtime_error("Space name must be one of l2 or ip");
        }
        alg_ = NULL;
        num_threads_default = std::thread::hardware_concurrency();
    }

    ~Index() {
        delete space;
        if (alg_) delete alg_;
    }

    void init_new_index(size_t dataset_size, size_t max_neighbors, size_t random_seed) {
        if (alg_) {
            throw std::runtime_error("The index is already initiated.");
        }
        alg_ = new GraphHierarchy(dataset_size, space, max_neighbors);
        alg_->random_seed_ = random_seed;
        cur_l = 0;
    }

    void addItems(py::object input, py::object ids_ = py::none(), int num_threads = -1) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
        auto buffer = items.request();
        if (num_threads <= 0) num_threads = num_threads_default;

        // check the dimensions of the input
        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);
        if (features != dimension) throw std::runtime_error("Wrong dimensionality of the vectors");
        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        // add the elements to the index
        {
            py::gil_scoped_release l;
            int start = 0;
            ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                alg_->addPoint((float*) items.data(row), (uint) id);
                });
            cur_l += rows;
        }
    }

    void load(std::string path) {
        if (alg_)
            alg_->loadGraph(path, dimension);
    }

    void save(std::string path) {
        if (alg_)
            alg_->saveGraph(path, dimension);
    }

    void set_construction_beam_size(size_t beam_size) {
      if (alg_)
          alg_->beamSizeConstruction_ = beam_size;
    }


    // build the approximate graph
    void build(int s, int p) { 
        alg_->buildGraph(s, p);
    }

    void construct_partitioning(int s) { 
        alg_->constructPartitioning(s);
    }


    void set_beam_size(size_t beam_size) {
      if (alg_)
          alg_->search_beam_size_ = beam_size;
    }

    void set_search_neighbors(size_t max_neighbors) {
      if (alg_)
          alg_->search_max_neighbors_ = max_neighbors;
    }

    // the search function
    py::object search(py::object input, size_t k = 1, int num_threads = -1) {
        py::array_t <float, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        uint* data_numpy_l;
        float* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0) num_threads = num_threads_default;
        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            // preparing output
            data_numpy_l = new uint[rows * k];
            data_numpy_d = new float[rows * k];

            // perform the search (in parallel)
            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<float,uint>> result = alg_->search((float*) items.data(row), k);
                while (result.size() < k) {
                    result.emplace(10000.0, 1);
                }
                if (result.size() != k)
                    throw std::runtime_error(
                        "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                for (int i = k - 1; i >= 0; i--) {
                    auto& result_tuple = result.top();
                    data_numpy_d[row * k + i] = result_tuple.first;
                    data_numpy_l[row * k + i] = result_tuple.second;
                    result.pop();
                }
            });
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<uint>(
                { rows, k },  // shape
                { k * sizeof(uint), sizeof(uint) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<float>(
                { rows, k },  // shape
                { k * sizeof(float), sizeof(float) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }

    // the search function
    py::object search_brute_force(py::object input, size_t k = 1, int num_threads = -1) {
        py::array_t <float, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        uint* data_numpy_l;
        float* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0) num_threads = num_threads_default;
        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            // preparing output
            data_numpy_l = new uint[rows * k];
            data_numpy_d = new float[rows * k];

            // perform the search (in parallel)
            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<float,uint>> result = alg_->search_bruteForce((float*) items.data(row), k);
                if (result.size() != k)
                    throw std::runtime_error(
                        "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                for (int i = k - 1; i >= 0; i--) {
                    auto& result_tuple = result.top();
                    data_numpy_d[row * k + i] = result_tuple.first;
                    data_numpy_l[row * k + i] = result_tuple.second;
                    result.pop();
                }
            });
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<uint>(
                { rows, k },  // shape
                { k * sizeof(uint), sizeof(uint) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<float>(
                { rows, k },  // shape
                { k * sizeof(float), sizeof(float) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }
};


PYBIND11_PLUGIN(GraphHierarchy) {
    py::module m("GraphHierarchy");

    py::class_<Index>(m, "Index")
        .def(py::init<const std::string&, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index", &Index::init_new_index, py::arg("dataset_size"), py::arg("max_neighbors") = 32, py::arg("random_seed") = 100)
        .def("add_items", &Index::addItems, py::arg("data"), py::arg("ids") = py::none(), py::arg("num_threads") = -1)
        .def("save", &Index::save, py::arg("path"))
        .def("load", &Index::load, py::arg("path"))
        .def("set_beam_size_construction", &Index::set_construction_beam_size, py::arg("beam_size") = 10)
        .def("build", &Index::build, py::arg("s") = 10, py::arg("p") = 100)
        .def("construct_partitioning", &Index::construct_partitioning, py::arg("s") = 10)
        .def("set_beam_size", &Index::set_beam_size, py::arg("beam_size") = 10)
        .def("set_search_neighbors", &Index::set_search_neighbors, py::arg("m") = 0)
        .def("search",
            &Index::search,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1)
        .def("search_brute_force",
            &Index::search_brute_force,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1);
    return m.ptr();
}
