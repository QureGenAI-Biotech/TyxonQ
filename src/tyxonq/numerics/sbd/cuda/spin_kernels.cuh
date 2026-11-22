// 将既有 Torch 两步升降算符的 gather、符号乘法、scatter 合并。
// 索引和投影由 Python 原实现生成，本文件不改变 S² 的物理定义。
#pragma once
#include <array>
#include <vector>
#include <cstdint>

struct SpinTransitionView {
    const int64_t* source_rows;
    const int64_t* source_columns;
    const int64_t* target_rows;
    const int64_t* target_columns;
    const double* row_signs;
    const double* column_signs;
    size_t n_rows;
    size_t n_columns;
};

__global__ void spin_diagonal_kernel(const double* input, double* output, size_t size, double sz_square) {
    for (size_t index=blockIdx.x*size_t(blockDim.x)+threadIdx.x; index<size; index+=size_t(blockDim.x)*gridDim.x)
        output[index] = sz_square*input[index];
}

template<bool reverse>
__global__ void spin_transition_kernel(const double* input, double* output,
        SpinTransitionView transition, size_t original_columns, size_t intermediate_columns) {
    const size_t size = transition.n_rows*transition.n_columns;
    for (size_t index=blockIdx.x*size_t(blockDim.x)+threadIdx.x; index<size; index+=size_t(blockDim.x)*gridDim.x) {
        const size_t row = index/transition.n_columns, column = index%transition.n_columns;
        const size_t source = transition.source_rows[row]*intermediate_columns+transition.source_columns[column];
        const size_t target = transition.target_rows[row]*original_columns+transition.target_columns[column];
        const double phase = transition.row_signs[row]*transition.column_signs[column];
        if constexpr (reverse)
            atomicAdd(output+target, 0.5*input[source]*phase);
        else
            atomicAdd(output+source, input[target]*phase);
    }
}

class FusedSpinContext {
    std::vector<SpinTransitionView> plus_, minus_;
    std::pair<size_t,size_t> plus_shape_, minus_shape_, shape_;
    double sz_square_;
    int device_;
    double* intermediate_ = nullptr;
    size_t capacity_ = 0;

    static std::vector<SpinTransitionView> views(const std::vector<std::array<uintptr_t,8>>& packed) {
        std::vector<SpinTransitionView> result;
        for (const auto& item : packed) {
            if (item[6] == 0 || item[7] == 0) continue;
            result.push_back({reinterpret_cast<const int64_t*>(item[0]), reinterpret_cast<const int64_t*>(item[1]),
                reinterpret_cast<const int64_t*>(item[2]), reinterpret_cast<const int64_t*>(item[3]),
                reinterpret_cast<const double*>(item[4]), reinterpret_cast<const double*>(item[5]), item[6], item[7]});
        }
        return result;
    }
    static unsigned blocks(size_t count) { return static_cast<unsigned>(std::min(size_t(65535),(count+255)/256)); }
    void apply(const double* input, double* output, const std::vector<SpinTransitionView>& transitions,
               std::pair<size_t,size_t> intermediate_shape, cudaStream_t stream) {
        if (transitions.empty()) return;
        checked(cudaMemsetAsync(intermediate_, 0, intermediate_shape.first*intermediate_shape.second*sizeof(double), stream));
        for (const auto& transition : transitions)
            spin_transition_kernel<false><<<blocks(transition.n_rows*transition.n_columns),256,0,stream>>>(
                input, intermediate_, transition, shape_.second, intermediate_shape.second);
        for (const auto& transition : transitions)
            spin_transition_kernel<true><<<blocks(transition.n_rows*transition.n_columns),256,0,stream>>>(
                intermediate_, output, transition, shape_.second, intermediate_shape.second);
    }
public:
    FusedSpinContext(const std::vector<std::array<uintptr_t,8>>& plus,
                     std::pair<size_t,size_t> plus_shape,
                     const std::vector<std::array<uintptr_t,8>>& minus,
                     std::pair<size_t,size_t> minus_shape,
                     std::pair<size_t,size_t> shape, double sz_square, int device, double memory_fraction)
        : plus_(views(plus)), minus_(views(minus)), plus_shape_(plus_shape), minus_shape_(minus_shape),
          shape_(shape), sz_square_(sz_square), device_(device) {
        if (!shape.first || !shape.second || !(memory_fraction>0 && memory_fraction<=1))
            throw std::invalid_argument("invalid fused spin shape or memory fraction");
        checked(cudaSetDevice(device_));
        capacity_ = std::max(plus_shape.first*plus_shape.second, minus_shape.first*minus_shape.second);
        size_t free=0, total=0; checked(cudaMemGetInfo(&free,&total));
        if (capacity_*sizeof(double)>free || total-free+capacity_*sizeof(double)>total*memory_fraction)
            throw std::runtime_error("fused spin CUDA memory budget exceeded");
        // 元数据张量由 Python contractor 持有；构造后地址不可更换。
        checked(cudaDeviceSynchronize());
        if (capacity_) checked(cudaMalloc(&intermediate_,capacity_*sizeof(double)));
    }
    ~FusedSpinContext() { cudaSetDevice(device_); if (intermediate_) cudaFree(intermediate_); }
    size_t memory_bytes() const { return capacity_*sizeof(double); }
    void contract(uintptr_t input, uintptr_t output, size_t size, uintptr_t stream_pointer) {
        if (size!=shape_.first*shape_.second) throw std::invalid_argument("fused spin coefficient size mismatch");
        checked(cudaSetDevice(device_));
        const auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
        for (auto pointer : {input,output}) {
            cudaPointerAttributes attributes; checked(cudaPointerGetAttributes(&attributes,reinterpret_cast<void*>(pointer)));
            if (attributes.type!=cudaMemoryTypeDevice || attributes.device!=device_)
                throw std::invalid_argument("fused spin pointer device mismatch");
        }
        py::gil_scoped_release release;
        checked(cudaStreamSynchronize(stream));
        const auto* c = reinterpret_cast<const double*>(input);
        auto* result = reinterpret_cast<double*>(output);
        spin_diagonal_kernel<<<blocks(size),256,0,stream>>>(c,result,size,sz_square_);
        apply(c,result,plus_,plus_shape_,stream);
        apply(c,result,minus_,minus_shape_,stream);
        checked(cudaGetLastError());
        // 一个 contract 只在边界同步，不在每个轨道之间同步。
        checked(cudaStreamSynchronize(stream));
    }
};

static void bind_spin_kernels(py::module_& module) {
    py::class_<FusedSpinContext>(module,"FusedSpinContext")
        .def(py::init<const std::vector<std::array<uintptr_t,8>>&,std::pair<size_t,size_t>,
             const std::vector<std::array<uintptr_t,8>>&,std::pair<size_t,size_t>,
             std::pair<size_t,size_t>,double,int,double>(),
             py::arg("plus"),py::arg("plus_shape"),py::arg("minus"),py::arg("minus_shape"),
             py::arg("shape"),py::arg("sz_square"),py::arg("device_id"),py::arg("memory_fraction")=0.8)
        .def("contract",&FusedSpinContext::contract)
        .def("memory_bytes",&FusedSpinContext::memory_bytes);
}
