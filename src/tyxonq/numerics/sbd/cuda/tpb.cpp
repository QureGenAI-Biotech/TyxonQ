// 为 SBD TPB 提供积分更新、Hamiltonian 算符像、RDM 和本征求解的内存接口。
#include <iostream>
#include <limits>
#include <cmath>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>
#include "sbd/caop/inc_all.h"
#include "sbd/chemistry/basic/inc_all.h"
#include "sbd/chemistry/tpb/inc_all.h"

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Labels = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using Device = thrust::device_vector<double>;
using Dets = sbd::det_vector<size_t, sbd::det_kind::half>;

static void checked(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
#include "spin_kernels.cuh"
static py::array_t<double> array_copy(const std::vector<double>& data) {
    py::array_t<double> out(data.size());
    std::copy(data.begin(), data.end(), out.mutable_data());
    return out;
}
static py::array_t<double> blocks_copy(const std::vector<std::vector<double>>& data) {
    const size_t width = data.empty() ? 0 : data[0].size();
    py::array_t<double> out({py::ssize_t(data.size()), py::ssize_t(width)});
    for (size_t i = 0; i < data.size(); ++i)
        std::copy(data[i].begin(), data[i].end(), out.mutable_data() + i * width);
    return out;
}

// 计数真实的 Davidson 算符作用次数，不猜测迭代是否收敛。
class CountedMult : public sbd::MultTPBThrust<double> {
public:
    size_t calls = 0;
    void run(const Device& h, const Device& c, Device& hc) override {
        ++calls;
        sbd::MultTPBThrust<double>::run(h, c, hc);
    }
};

class TPBContext {
    const size_t bit_length_ = 20;
    int norb_, device_, rank_ = 0, size_ = 1;
    std::pair<int, int> nelec_, grid_;
    double memory_fraction_;
    size_t na_, nb_, a0_, a1_, b0_, b1_, local_size_;
    MPI_Comm comm_ = MPI_COMM_NULL, h_ = MPI_COMM_NULL, b_ = MPI_COMM_NULL;
    MPI_Comm t_ = MPI_COMM_NULL, a_ = MPI_COMM_NULL;
    std::unique_ptr<CountedMult> mult_;
    Device diagonal_, input_, output_;

    void free_communicators() noexcept {
        int finalized = 0; MPI_Finalized(&finalized);
        if (!finalized) for (auto p : {&a_, &t_, &b_, &h_, &comm_})
            if (*p != MPI_COMM_NULL) MPI_Comm_free(p);
    }

    static std::vector<std::vector<size_t>> unpack(const Labels& labels, int n, int electrons) {
        if (labels.ndim() != 1 || labels.size() == 0)
            throw std::invalid_argument("determinant labels must be a nonempty 1D array");
        std::vector<std::vector<size_t>> out;
        uint64_t previous = 0;
        for (py::ssize_t i = 0; i < labels.size(); ++i) {
            const int64_t signed_value = labels.data()[i];
            const uint64_t value = static_cast<uint64_t>(signed_value);
            if (signed_value < 0 || (value >> n) || __builtin_popcountll(value) != electrons)
                throw std::invalid_argument("invalid determinant label or electron count");
            if (i && value <= previous)
                throw std::invalid_argument("determinant labels must be sorted and unique");
            previous = value;
            std::vector<size_t> words((n + 19) / 20, 0);
            for (size_t w = 0; w < words.size(); ++w)
                words[w] = (value >> (20 * w)) & ((size_t(1) << 20) - 1);
            out.push_back(std::move(words));
        }
        return out;
    }

    void activate() const { checked(cudaSetDevice(device_)); }
    void memory_guard(size_t additional, const char* stage) const {
        size_t free = 0, total = 0;
        checked(cudaMemGetInfo(&free, &total));
        int failed = additional > free || total - free + additional > memory_fraction_ * total;
        // 任一 rank 预算不足时，所有 rank 一致退出，避免其他进程卡在集体操作。
        int any_failed = 0;
        MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm_);
        if (any_failed) throw std::runtime_error(std::string("SBD CUDA memory budget exceeded before ") + stage);
    }
    void require_serial(size_t count) const {
        if (size_ != 1) throw std::invalid_argument("device pointer operations require COMM_SELF");
        if (count != local_size_) throw std::invalid_argument("coefficient size mismatch");
    }
    void copy_input(uintptr_t ptr, size_t count) {
        require_serial(count);
        activate();
        cudaPointerAttributes attributes;
        checked(cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(ptr)));
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != device_)
            throw std::invalid_argument("input pointer must refer to the context CUDA device");
        // 与 PyTorch 的非默认流同步；其后仅发生同卡设备复制。
        checked(cudaDeviceSynchronize());
        checked(cudaMemcpy(thrust::raw_pointer_cast(input_.data()), reinterpret_cast<void*>(ptr),
                           count * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    void integrals(const Array& h1, const Array& eri, sbd::oneInt<double>& i1,
                   sbd::twoInt<double>& i2) const {
        if (h1.ndim() != 2 || h1.shape(0) != norb_ || h1.shape(1) != norb_ ||
            eri.ndim() != 4 || eri.shape(0) != norb_ || eri.shape(1) != norb_ ||
            eri.shape(2) != norb_ || eri.shape(3) != norb_)
            throw std::invalid_argument("expected h1(n,n) and eri(n,n,n,n)");
        const auto h = h1.unchecked<2>();
        const auto g = eri.unchecked<4>();
        i1.norbs = 2 * norb_;
        i1.store.assign(4 * norb_ * norb_, 0.0);
        const size_t pairs = norb_ * (norb_ + 1) / 2;
        i2.norbs = norb_;
        i2.store.assign(pairs * (pairs + 1) / 2, 0.0);
        for (int p = 0; p < norb_; ++p) for (int q = 0; q < norb_; ++q) {
            if (!std::isfinite(h(p,q))) throw std::invalid_argument("nonfinite one-body integral");
            i1(2*p,2*q) = i1(2*p+1,2*q+1) = h(p,q);
            for (int r = 0; r <= p; ++r) for (int s = 0; s < norb_; ++s) {
                if (!std::isfinite(g(p,q,r,s))) throw std::invalid_argument("nonfinite two-body integral");
                i2(2*p,2*q,2*r,2*s) = g(p,q,r,s);
            }
        }
        i2.DirectMat.resize(4 * norb_ * norb_);
        i2.ExchangeMat.resize(4 * norb_ * norb_);
        for (int p = 0; p < 2*norb_; ++p) for (int q = 0; q < 2*norb_; ++q) {
            i2.Direct(p,q) = i2.Value(p,p,q,q);
            i2.Exchange(p,q) = i2.Value(p,q,q,p);
        }
    }
    std::vector<double> gather(const std::vector<double>& local) const {
        if (na_ * nb_ > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("MPI coefficient count exceeds INT_MAX");
        std::vector<double> packed(na_ * nb_, 0.0), full;
        for (size_t ia = a0_; ia < a1_; ++ia)
            for (size_t ib = b0_; ib < b1_; ++ib)
                packed[ia*nb_+ib] = local[(ia-a0_)*(b1_-b0_)+ib-b0_];
        if (rank_ == 0) full.resize(packed.size());
        MPI_Reduce(packed.data(), full.data(), static_cast<int>(packed.size()), MPI_DOUBLE,
                   MPI_SUM, 0, comm_);
        return full;
    }
public:
    TPBContext(const Array& h1, const Array& eri, const Labels& sa, const Labels& sb,
               int n, std::pair<int,int> electrons, int device, py::object comm,
               std::pair<int,int> grid, double memory_fraction)
        : norb_(n), device_(device), nelec_(electrons), grid_(grid),
          memory_fraction_(memory_fraction), na_(sa.size()), nb_(sb.size()) {
        if (n < 1 || n > 63) throw std::invalid_argument("TPB int64 labels require 1 <= norb <= 63");
        if (!(memory_fraction > 0 && memory_fraction <= 1))
            throw std::invalid_argument("memory_fraction must lie in (0, 1]");
        activate();
        int initialized = 0; MPI_Initialized(&initialized);
        if (!initialized) throw std::runtime_error("MPI must be initialized by mpi4py");
        MPI_Comm parent = MPI_COMM_SELF;
        if (!comm.is_none()) {
            auto ptr = PyMPIComm_Get(comm.ptr());
            if (!ptr) throw py::error_already_set();
            parent = *ptr;
        }
        MPI_Comm_size(parent, &size_); MPI_Comm_rank(parent, &rank_);
        if (grid.first < 1 || grid.second < 1 || grid.first*grid.second != size_ ||
            size_t(grid.first) > na_ || size_t(grid.second) > nb_)
            throw std::invalid_argument("grid must match communicator size and nonempty determinant blocks");
        auto av = unpack(sa, n, electrons.first), bv = unpack(sb, n, electrons.second);
        // 上游容器宽度是静态量；构造阶段受 GIL 保护，之后 GPU 数据独立持有。
        Dets::tyxonq_set_elem_size((n + 19) / 20);
        sbd::det_vector<size_t>::tyxonq_set_elem_size((2*n + 19) / 20);
        Dets adets(av.begin(), av.end()), bdets(bv.begin(), bv.end());
        sbd::oneInt<double> i1; sbd::twoInt<double> i2;
        integrals(h1, eri, i1, i2);
        MPI_Comm_dup(parent, &comm_);
        try {
        sbd::TaskCommunicator(comm_, 1, grid.first, grid.second, 1, h_, b_, t_, a_);
        std::vector<sbd::TaskHelpers> helpers;
        sbd::MakeHelpers(adets, bdets, bit_length_, n, helpers, h_, b_, t_, grid.first, grid.second);
        sbd::RemakeHelpers(adets, bdets, bit_length_, n, helpers, h_, b_, t_, grid.first, grid.second);
        if (helpers.empty()) throw std::runtime_error("empty TPB task table");
        a0_ = helpers[0].braAlphaStart; a1_ = helpers[0].braAlphaEnd;
        b0_ = helpers[0].braBetaStart; b1_ = helpers[0].braBetaEnd;
        local_size_ = (a1_-a0_)*(b1_-b0_);
        size_t connection_bytes = 0, max_a = 0, max_b = 0;
        for (const auto& helper : helpers) {
            const size_t a = helper.braAlphaEnd-helper.braAlphaStart;
            const size_t b = helper.braBetaEnd-helper.braBetaStart;
            max_a = std::max(max_a,a); max_b = std::max(max_b,b);
            if (helper.taskType != 1) {
                connection_bytes += (a+1)*sizeof(size_t);
                for (size_t i=0; i<a; ++i)
                    connection_bytes += helper.SinglesFromAlphaLen[i]*(sizeof(size_t)+2*sizeof(int));
                if (helper.taskType == 2) {
                    connection_bytes += (a+1)*sizeof(size_t);
                    for (size_t i=0; i<a; ++i)
                        connection_bytes += helper.DoublesFromAlphaLen[i]*(sizeof(size_t)+4*sizeof(int));
                }
            }
            if (helper.taskType != 2) {
                connection_bytes += (b+1)*sizeof(size_t);
                for (size_t i=0; i<b; ++i)
                    connection_bytes += helper.SinglesFromBetaLen[i]*(sizeof(size_t)+2*sizeof(int));
                if (helper.taskType == 1) {
                    connection_bytes += (b+1)*sizeof(size_t);
                    for (size_t i=0; i<b; ++i)
                        connection_bytes += helper.DoublesFromBetaLen[i]*(sizeof(size_t)+4*sizeof(int));
                }
            }
        }
        const size_t integral_bytes = (i1.store.size()+i2.store.size()+i2.DirectMat.size()+i2.ExchangeMat.size())*sizeof(double);
        const size_t determinant_bytes = (((n+19)/20)*(na_+nb_)+((2*n+19)/20)*max_a*max_b)*sizeof(size_t);
        memory_guard(connection_bytes+integral_bytes+determinant_bytes+5*local_size_*sizeof(double), "TPB context allocation");
        mult_ = std::make_unique<CountedMult>();
        mult_->Init(adets, bdets, bit_length_, n, grid.first, grid.second, helpers,
                    0.0, i1, i2, h_, b_, t_, a_, true, -1, false);
        const size_t largest_block = ((na_+grid.first-1)/grid.first)*((nb_+grid.second-1)/grid.second);
        for (auto& buffer : mult_->transfer_workspace) buffer.reserve(largest_block);
        mult_->makeQChamDiagTerms(diagonal_);
        input_.resize(local_size_); output_.resize(local_size_);
        checked(cudaDeviceSynchronize());
        } catch (...) {
            // 构造失败时析构函数不会执行，因此显式释放已建立的通信器。
            free_communicators();
            throw;
        }
    }
    ~TPBContext() {
        cudaSetDevice(device_);
        mult_.reset();
        free_communicators();
    }
    void update_integrals(const Array& h1, const Array& eri) {
        activate();
        sbd::oneInt<double> i1; sbd::twoInt<double> i2;
        integrals(h1, eri, i1, i2);
        // 积分更新不重建与组态集合有关的连接表。
        thrust::copy(i1.store.begin(), i1.store.end(), mult_->I1_store.begin());
        thrust::copy(i2.store.begin(), i2.store.end(), mult_->I2_store.begin());
        thrust::copy(i2.DirectMat.begin(), i2.DirectMat.end(), mult_->I2_dm.begin());
        thrust::copy(i2.ExchangeMat.begin(), i2.ExchangeMat.end(), mult_->I2_em.begin());
        mult_->makeQChamDiagTerms(diagonal_);
        checked(cudaDeviceSynchronize());
    }
    py::array_t<double> hdiag() {
        activate();
        std::vector<double> local(local_size_);
        thrust::copy(diagonal_.begin(), diagonal_.end(), local.begin());
        return array_copy(gather(local));
    }
    size_t memory_bytes() const {
        size_t count = (diagonal_.capacity()+input_.capacity()+output_.capacity())*sizeof(double);
        count += (mult_->adets.capacity()+mult_->bdets.capacity()+mult_->dets.capacity())*sizeof(size_t);
        count += (mult_->I1_store.capacity()+mult_->I2_store.capacity()+mult_->I2_dm.capacity()+mult_->I2_em.capacity())*sizeof(double);
        for (const auto& values : mult_->helper_storage) count += values.capacity()*sizeof(size_t);
        for (const auto& values : mult_->CrAn_storage) count += values.capacity()*sizeof(int);
        for (const auto& values : mult_->transfer_workspace) count += values.capacity()*sizeof(double);
        return count;
    }
    void matvec_device(uintptr_t in, uintptr_t out, size_t count) {
        copy_input(in, count);
        cudaPointerAttributes attrs; checked(cudaPointerGetAttributes(&attrs, reinterpret_cast<void*>(out)));
        if (attrs.type != cudaMemoryTypeDevice || attrs.device != device_)
            throw std::invalid_argument("output pointer must refer to the context CUDA device");
        { py::gil_scoped_release release;
          thrust::fill(output_.begin(), output_.end(), 0.0);
          mult_->run(diagonal_, input_, output_);
          checked(cudaMemcpy(reinterpret_cast<void*>(out), thrust::raw_pointer_cast(output_.data()),
                             count*sizeof(double), cudaMemcpyDeviceToDevice));
          checked(cudaDeviceSynchronize()); }
    }
    py::dict rdms_device(uintptr_t in, size_t count) {
        copy_input(in, count);
        const size_t n = norb_;
        memory_guard((2*n*n+4*n*n*n*n+2*local_size_)*sizeof(double), "RDM allocation");
        std::vector<std::vector<double>> dm1, dm2;
        { py::gil_scoped_release release;
          mult_->correlation_device(input_, dm1, dm2); checked(cudaDeviceSynchronize()); }
        py::dict out; out["one_p_rdm"] = blocks_copy(dm1); out["two_p_rdm"] = blocks_copy(dm2);
        return out;
    }
    py::dict solve(double tolerance, int max_cycles, int max_space, py::object initial, bool compute_rdms) {
        if (!(tolerance > 0) || max_cycles < 1 || max_space < 2)
            throw std::invalid_argument("invalid Davidson convergence settings");
        activate();
        const size_t n = norb_;
        const size_t subspace = std::min(size_t(max_space),na_*nb_);
        const size_t solver_bytes = (2*subspace+8)*local_size_*sizeof(double);
        const size_t rdm_bytes = compute_rdms ? (2*n*n+4*n*n*n*n+4*local_size_)*sizeof(double) : 0;
        memory_guard(std::max(solver_bytes,rdm_bytes), "Davidson/RDM allocation");
        std::vector<double> w(local_size_, 0.0);
        if (initial.is_none()) {
            // 无 HF 组态时仍有合法初猜；和标签原顺序一致。
            if (a0_ == 0 && b0_ == 0) w[0] = 1.0;
        } else {
            Array guess = py::cast<Array>(initial);
            if (size_t(guess.size()) != na_*nb_) throw std::invalid_argument("initial coefficient size mismatch");
            for (size_t ia=a0_; ia<a1_; ++ia) for (size_t ib=b0_; ib<b1_; ++ib)
                w[(ia-a0_)*(b1_-b0_)+ib-b0_] = guess.data()[ia*nb_+ib];
        }
        double energy = 0, residual = 0, norm = 0;
        double davidson_seconds = 0, residual_seconds = 0, rdm_seconds = 0;
        std::vector<std::vector<double>> dm1, dm2;
        std::vector<double> hamiltonian_image;
        size_t iterations;
        { py::gil_scoped_release release;
          sbd::InnerProduct(w,w,norm,b_);
          if (!(norm > 0) || !std::isfinite(norm)) throw std::invalid_argument("invalid initial vector norm");
          for (double& x : w) x /= std::sqrt(norm);
          mult_->calls = 0;
          const auto davidson_start = std::chrono::steady_clock::now();
          sbd::Davidson(diagonal_, w, *mult_, max_cycles,
                        std::min(max_space, static_cast<int>(na_*nb_)), tolerance, 600.0);
          checked(cudaDeviceSynchronize());
          const auto residual_start = std::chrono::steady_clock::now();
          davidson_seconds = std::chrono::duration<double>(residual_start-davidson_start).count();
          iterations = mult_->calls;
          thrust::copy(w.begin(), w.end(), input_.begin());
          thrust::fill(output_.begin(), output_.end(), 0.0);
          mult_->run(diagonal_, input_, output_);
          std::vector<double> hw(local_size_);
          thrust::copy(output_.begin(), output_.end(), hw.begin());
          hamiltonian_image = hw;
          sbd::InnerProduct(w,hw,energy,b_);
          sbd::InnerProduct(w,w,norm,b_);
          energy /= norm;
          for (size_t i=0; i<w.size(); ++i) hw[i] -= energy*w[i];
          sbd::InnerProduct(hw,hw,residual,b_);
          residual = std::sqrt(residual/norm);
          const auto rdm_start = std::chrono::steady_clock::now();
          residual_seconds = std::chrono::duration<double>(rdm_start-residual_start).count();
          if (compute_rdms) mult_->correlation_device(input_, dm1, dm2);
          checked(cudaDeviceSynchronize());
          rdm_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-rdm_start).count(); }
        auto coefficients = gather(w);
        auto global_hamiltonian_image = gather(hamiltonian_image);
        py::dict out;
        out["rank"] = rank_; out["world_size"] = size_;
        out["energy"] = energy; out["residual"] = residual;
        out["converged"] = std::isfinite(residual) && residual <= tolerance;
        out["iterations"] = iterations;
        out["h_matvecs"] = iterations + 1;
        py::dict timing;
        timing["davidson_seconds"] = davidson_seconds;
        timing["residual_seconds"] = residual_seconds;
        timing["rdm_seconds"] = rdm_seconds;
        out["timings"] = timing;
        out["context_memory_bytes"] = memory_bytes();
        if (rank_ == 0) {
            out["coefficients"] = array_copy(coefficients);
            out["hamiltonian_image"] = array_copy(global_hamiltonian_image);
            if (compute_rdms) {
                out["one_p_rdm"] = blocks_copy(dm1); out["two_p_rdm"] = blocks_copy(dm2);
            } else {
                out["one_p_rdm"] = py::none(); out["two_p_rdm"] = py::none();
            }
        } else {
            out["coefficients"] = py::none(); out["one_p_rdm"] = py::none(); out["two_p_rdm"] = py::none();
            out["hamiltonian_image"] = py::none();
        }
        return out;
    }
};

PYBIND11_MODULE(_sbd_tpb_cuda, module) {
    if (import_mpi4py() < 0) throw py::error_already_set();
    module.attr("core_commit") = "93ebabec6c841d0ade706018b053795d8c1db9bd";
    module.attr("non_cuda_aware_mpi") = true;
    bind_spin_kernels(module);
    py::class_<TPBContext>(module, "TPBContext")
        .def(py::init<const Array&,const Array&,const Labels&,const Labels&,int,std::pair<int,int>,int,py::object,std::pair<int,int>,double>(),
             py::arg("h1e"),py::arg("eri"),py::arg("strings_a"),py::arg("strings_b"),
             py::arg("norb"),py::arg("nelec"),py::arg("device_id")=0,
             py::arg("comm")=py::none(),py::arg("grid")=std::make_pair(1,1),py::arg("memory_fraction")=0.8)
        .def("update_integrals", &TPBContext::update_integrals)
        .def("hdiag", &TPBContext::hdiag)
        .def("memory_bytes", &TPBContext::memory_bytes)
        .def("matvec_device", &TPBContext::matvec_device)
        .def("rdms_device", &TPBContext::rdms_device)
        .def("solve", &TPBContext::solve,py::arg("tolerance"),py::arg("max_cycles"),
             py::arg("max_space"),py::arg("initial_coefficients")=py::none(),py::arg("compute_rdms")=true);
}
