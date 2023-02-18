#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define sleep(t) std::this_thread::sleep_for(std::chrono::milliseconds(t))
#define read_and_shift(arg_str, cnt) \
    std::string arg_str;             \
    if (cnt < argc) arg_str = argv[cnt++]

const float bytes_per_gb = (1 << 30);
const float ms_per_hour = 1000 * 3600;
const int max_grid_dim = (1 << 15);
const int max_block_dim = 1024;
const int max_sleep_time = 1e3;
const float sleep_interval = 1e16;
const int max_gpu_num = 32;

__global__ void default_script_kernel(char* array, size_t occupy_size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= occupy_size) return;
    array[i]++;
}

void launch_default_script(char** array, size_t occupy_size,
                           std::vector<int>& grid_dim,
                           std::vector<int>& gpu_ids) {
    int gd = std::min(grid_dim[rand() % grid_dim.size()],
                      int(occupy_size / max_block_dim));
    for (int id : gpu_ids) {
        cudaSetDevice(id);
        default_script_kernel<<<gd, max_block_dim, 0, NULL>>>(array[id],
                                                              occupy_size);
    }
}

void run_default_script(char** array, size_t occupy_size, float total_time,
                        std::vector<int>& gpu_ids) {
    // printf("Running default script >>>>>>>>>>>>>>>>>>>>\n");
    srand(time(NULL));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
    std::vector<int> grid_dim;
    for (int i = 1; i <= max_grid_dim; i <<= 1) {
        grid_dim.push_back(i);
    }
    cudaEventRecord(start, 0);

    // std::time_t now = std::time(0);
    // tm* localtm = localtime(&now);
    // std::cout << "Occupied since local time: " << asctime(localtm) << std::endl;
    int cnt = 0;
    while (true) {
        launch_default_script(array, occupy_size, grid_dim, gpu_ids);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        if (total_time >= 0 && time / ms_per_hour > total_time) break;
        if (!((++cnt) % size_t(sleep_interval / occupy_size))) {
            cnt = 0;
            // printf("Occupied time: %.2f hours\n", time / ms_per_hour);
            int sleep_time = rand() % max_sleep_time + 1;
            sleep(sleep_time);
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void process_args(int argc, char** argv, std::vector<int>& gpu_ids, size_t& occupy_size, float& total_time, std::string& script_path, int& mode) {
    int cnt = 13;

    int gpu_num;
    cudaGetDeviceCount(&gpu_num);
    srand(time(NULL));

    std::string id_str = argv[cnt++];

    std::replace(id_str.begin(), id_str.end(), ',', ' ');
    std::stringstream ss;
    ss << id_str;

    int id;
    while (ss >> id) {
        gpu_ids.push_back(id);
    }
    if (gpu_ids.size() == 1 && gpu_ids[0] == -1) {
        gpu_ids[0] = 0;
        for (int i = 1; i < gpu_num; ++i) {
            gpu_ids.push_back(i);
        }
    }

    mode = 0;
    if (cnt < argc) {
        sscanf(argv[cnt++], "%d", &mode);
    }

    total_time = -1;
    if (cnt < argc) {
        sscanf(argv[cnt++], "%f", &total_time);
    }

    float occupy_mem = -1;
    if (cnt < argc) {
        sscanf(argv[cnt++], "%f", &occupy_mem);
        occupy_size = size_t(occupy_mem * bytes_per_gb);
    }

    float random_left = 1;
    if (cnt < argc) {
        sscanf(argv[cnt++], "%f", &random_left);
    }

    if (occupy_mem == -1) {
        cudaSetDevice(gpu_ids[0]);
        size_t total_size, avail_size;
        cudaMemGetInfo(&avail_size, &total_size);
        occupy_size = total_size - size_t(bytes_per_gb * random_left);
    }

    script_path = "";
    if (cnt < argc) {
        script_path = argv[cnt++];
    }
}

void allocate_mem(char** array, size_t occupy_size, std::vector<int>& gpu_ids, int mode) {
    std::vector<size_t> allocated(max_gpu_num, 0);
    while (true) {
        // printf("Try allocate GPU memory %d times >>>>>>>>>>>>>>>>>>>>\n", ++cnt);
        int num_allocated = 0;
        for (int id : gpu_ids) {
            if (allocated[id] != occupy_size) {
                cudaSetDevice(id);
                size_t total_size, avail_size;
                cudaMemGetInfo(&avail_size, &total_size);
                size_t target_size = 0;

                if (mode != 2) {  // fight and occupy/release
                    target_size = min(avail_size - size_t(bytes_per_gb * 0.1), occupy_size - allocated[id]);
                } else {
                    // peace mode
                    // first occupy: av > 5G
                    // consecutive: wait until all finish
                    if (allocated[id] == 0) {
                        if (avail_size > size_t(5 * bytes_per_gb)) {
                            target_size = min(avail_size - size_t(bytes_per_gb * 2), occupy_size);
                        } else
                            target_size = 0;
                    } else {
                        target_size = occupy_size - allocated[id];
                    }
                }

                cudaError_t status = cudaMalloc(&array[id], target_size);
                if (status == cudaSuccess) {
                    allocated[id] += target_size;
                    cudaMemGetInfo(&avail_size, &total_size);
                    if (target_size >= size_t(bytes_per_gb * 0.5)) {
                        printf(
                            "[RUNNING] GPU-%d: Successfully allocate %.2f GB GPU memory (%.2f GB "
                            "available)\n",
                            id, target_size / bytes_per_gb, avail_size / bytes_per_gb);
                    }
                }
                if (allocated[id] == occupy_size) {
                    num_allocated++;
                    cudaMemGetInfo(&avail_size, &total_size);
                    printf(
                        "[DONE] GPU-%d: Successfully allocate %.2f GB GPU memory (%.2f GB "
                        "available)\n",
                        id, occupy_size / bytes_per_gb, avail_size / bytes_per_gb);
                }
            }
        }
        // run_default_script(array, occupy_size, 2e-4, gpu_ids);
        if (num_allocated == gpu_ids.size()) break;
    }
    sleep(500);
    // inform_email(gpu_ids);
    // printf("Successfully allocate memory on all GPUs!\n");
}

void run_custom_script(char** array, std::vector<int>& gpu_ids,
                       std::string script_path) {
    printf("Running custom script >>>>>>>>>>>>>>>>>>>>\n");
    for (int id : gpu_ids) {
        cudaFree(array[id]);
    }
    std::string cmd = "sh " + script_path;
    std::system(cmd.c_str());
}

int main(int argc, char** argv) {
    size_t occupy_size;
    float total_time;
    std::vector<int> gpu_ids;
    std::string script_path;
    char* array[max_gpu_num];
    int mode;

    process_args(argc, argv, gpu_ids, occupy_size, total_time, script_path, mode);
    // mode = 0: fight and occupy
    // mode = 1: fight and exit
    // mode = 2: peace

    if (mode == 1) total_time = 0.0;

    allocate_mem(array, occupy_size, gpu_ids, mode);

    if (script_path == "") {
        run_default_script(array, occupy_size, total_time, gpu_ids);
        for (int id : gpu_ids) {
            cudaFree(array[id]);
        }
    } else {
        run_custom_script(array, gpu_ids, script_path);
    }

    return 0;
}
