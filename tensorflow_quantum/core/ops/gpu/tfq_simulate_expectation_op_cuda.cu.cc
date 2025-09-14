/* Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <vector>

#include <chrono>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/gates_qsim.h"
#include "../qsim/lib/seqfor.h"
#include "../qsim/lib/simulator_cuda.h"
#include "../qsim/lib/statespace_cuda.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/proto/program.pb.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::tensorflow::Status;
using ::tfq::proto::PauliSum;
using ::tfq::proto::Program;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;


Status AllocateQsimTempTensors(
    tensorflow::OpKernelContext* context, tensorflow::Tensor* d_wf_tensor,
     tensorflow::Tensor* d_idx_tensor, tensorflow::Tensor* d_ms_tensor,
     tensorflow::Tensor* d_xss_tensor) {
  tensorflow::AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(false);
  alloc_attr.set_gpu_compatible(true);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      tensorflow::DataType::DT_FLOAT,
      tensorflow::TensorShape({131072 * sizeof(float)}),
      d_wf_tensor, alloc_attr));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      tensorflow::DataType::DT_UINT32,
      tensorflow::TensorShape({992 * sizeof(unsigned)}),
      d_idx_tensor, alloc_attr));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      tensorflow::DataType::DT_UINT64,
      tensorflow::TensorShape({7 * sizeof(uint64_t)}),
      d_ms_tensor, alloc_attr));
  TF_RETURN_IF_ERROR(context->allocate_temp(
      tensorflow::DataType::DT_UINT64,
      tensorflow::TensorShape({64 * sizeof(uint64_t)}),
      d_xss_tensor, alloc_attr));
  return Status::OK();
}

// __global__ void ComputeSmallCudaKernel(const int total_size,
//     int output_dim_op_size, int* num_qubits,
//     const thrust::host_vector<thrust::host_vector<qsim::GateFused<QsimGate>>>& fused_circuits,
//     const thrust::host_vector<thrust::host_vector<PauliSum>>& pauli_sums,
//     float* out) {
//   int old_batch_index = -2;
//   int cur_batch_index = -1;
//   int largest_nq = 1;
//   int cur_op_index;
//   auto sv = ss.Create(largest_nq);
//   auto scratch = ss.Create(largest_nq);

//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size;
//        i += blockDim.x * gridDim.x) {
//     cur_batch_index = i / output_dim_op_size;
//     cur_op_index = i % output_dim_op_size;

//     const int nq = num_qubits[cur_batch_index];
//     // (#679) Just ignore empty program
//     if (fused_circuits[cur_batch_index].size() == 0) {
//       out[i] = -2.0;
//       continue;
//     }
//     if (cur_batch_index != old_batch_index) {
//       // We've run into a new state vector we must compute.
//       // Only compute a new state vector when we have to.
//       if (nq > largest_nq) {
//         largest_nq = nq;
//         sv = ss.Create(largest_nq);
//         scratch = ss.Create(largest_nq);
//       }
//       // no need to update scratch_state since ComputeExpectation
//       // will take care of things for us.
//       ss.SetStateZero(sv);
//       for (int j = 0; j < fused_circuits[cur_batch_index].size(); j++) {
//         qsim::ApplyFusedGate(sim, fused_circuits[cur_batch_index][j], sv);
//       }
//     }

//     float exp_v = 0.0;
//     ComputeExpectationQsim(pauli_sums[cur_batch_index][cur_op_index],
//                            sim, ss, sv, scratch, &exp_v),
//     out[i] = exp_v;
//     old_batch_index = cur_batch_index;
//   }
// }


class TfqSimulateExpectationOpGpuCpu : public tensorflow::OpKernel {
 public:
  explicit TfqSimulateExpectationOpGpuCpu(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    // Get the number of CPU cycle in ComputeSmall via attributes.
    OP_REQUIRES_OK(context, context->GetAttr("cpu_cycle", &cpu_cycle_));

    // Get the number of threads in SimulatorCUDA via attributes.
    OP_REQUIRES_OK(context, context->GetAttr("num_threads_in_sim",
                                             &num_threads_in_sim_));

    // Get the number of blocks & threads in StateSpaceCUDA.
    OP_REQUIRES_OK(context, context->GetAttr("block_count", &block_count_));
    OP_REQUIRES_OK(context, context->GetAttr("thread_per_block",
                                             &thread_per_block_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    // TODO (mbbrough): add more dimension checks for other inputs here.
    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 4,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 4 inputs, got ", num_inputs, " inputs.")));

    // Create the output Tensor.
    const int output_dim_batch_size = context->input(0).dim_size(0);
    const int output_dim_op_size = context->input(3).dim_size(1);
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(output_dim_batch_size);
    output_shape.AddDim(output_dim_op_size);

    tensorflow::Tensor* output = nullptr;
    tensorflow::AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    alloc_attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output,
                                                     alloc_attr));
    auto output_tensor = output->matrix<float>();
    // Parse program protos.
    std::vector<Program> programs;
    std::vector<int> num_qubits;
    std::vector<std::vector<PauliSum>> pauli_sums;
    OP_REQUIRES_OK(context, GetProgramsAndNumQubits(context, &programs,
                                                    &num_qubits, &pauli_sums));

    std::vector<SymbolMap> maps;
    OP_REQUIRES_OK(context, GetSymbolMaps(context, &maps));

    OP_REQUIRES(context, programs.size() == maps.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of circuits and symbol_values do not match. Got ",
                    programs.size(), " circuits and ", maps.size(),
                    " symbol values.")));

    // Construct qsim circuits.
    std::vector<QsimCircuit> qsim_circuits(programs.size(), QsimCircuit());
    std::vector<std::vector<qsim::GateFused<QsimGate>>> fused_circuits(
        programs.size(), std::vector<qsim::GateFused<QsimGate>>({}));

    Status parse_status = Status::OK();
    auto p_lock = tensorflow::mutex();
    auto construct_f = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        Status local =
            QsimCircuitFromProgram(programs[i], maps[i], num_qubits[i],
                                   &qsim_circuits[i], &fused_circuits[i]);
        NESTED_FN_STATUS_SYNC(parse_status, local, p_lock);
      }
    };

    const int num_cycles = 1000;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        programs.size(), num_cycles, construct_f);
    OP_REQUIRES_OK(context, parse_status);

    int max_num_qubits = 0;
    for (const int num : num_qubits) {
      max_num_qubits = std::max(max_num_qubits, num);
    }

    if (max_num_qubits >= 26 || programs.size() == 1) {
      tensorflow::Tensor d_wf_tensor;
      tensorflow::Tensor d_idx_tensor;
      tensorflow::Tensor d_ms_tensor;
      tensorflow::Tensor d_xss_tensor;
      OP_REQUIRES_OK(context, AllocateQsimTempTensors(
          context, &d_wf_tensor, &d_idx_tensor, &d_ms_tensor, &d_xss_tensor));
      ComputeLarge(num_qubits, fused_circuits, pauli_sums, context,
                   &output_tensor, d_wf_tensor.flat<float>().data(),
                   d_idx_tensor.flat<unsigned>().data(),
                   d_ms_tensor.flat<uint64_t>().data(),
                   d_xss_tensor.flat<uint64_t>().data());
    } else {
      ComputeSmall(num_qubits, max_num_qubits, fused_circuits, pauli_sums,
                   context, &output_tensor);
    }
  }

 private:
  int num_threads_in_sim_;
  int thread_per_block_;
  int block_count_;
  int cpu_cycle_;

  // Define the GPU implementation that launches the CUDA kernel.
  void ComputeLarge(
      const std::vector<int>& num_qubits,
      const std::vector<std::vector<qsim::GateFused<QsimGate>>>& fused_circuits,
      const std::vector<std::vector<PauliSum>>& pauli_sums,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<float, 1>::Matrix* output_tensor,
      float* d_wf, unsigned* d_idx, uint64_t* d_ms, uint64_t* d_xss) {
    // Instantiate qsim objects.
    using Simulator = qsim::SimulatorCUDA<float>;
    using StateSpace = Simulator::StateSpace;
    // Launch the cuda kernel.
    // TFQ GPU
    Simulator::Parameter param_sim;
    param_sim.num_threads = num_threads_in_sim_;

    StateSpace::Parameter param_ss;
    param_ss.num_threads = thread_per_block_;
    param_ss.num_dblocks = block_count_;

    // Begin simulation.
    int largest_nq = 1;
    Simulator sim = Simulator(param_sim, d_wf, d_idx, d_ms, d_xss);
    StateSpace ss = StateSpace(param_ss);
    auto sv = ss.Create(largest_nq);
    auto scratch = ss.Create(largest_nq);

    // Simulate programs one by one. Parallelizing over state vectors
    // we no longer parallelize over circuits. Each time we encounter a
    // a larger circuit we will grow the Statevector as necessary.
    for (int i = 0; i < fused_circuits.size(); i++) {
      int nq = num_qubits[i];

      if (nq > largest_nq) {
        // need to switch to larger statespace.
        largest_nq = nq;
        sv = ss.Create(largest_nq);
        scratch = ss.Create(largest_nq);
      }
      // TODO: add heuristic here so that we do not always recompute
      //  the state if there is a possibility that circuit[i] and
      //  circuit[i + 1] produce the same state.
      ss.SetStateZero(sv);
      for (int j = 0; j < fused_circuits[i].size(); j++) {
        qsim::ApplyFusedGate(sim, fused_circuits[i][j], sv);
      }
      for (int j = 0; j < pauli_sums[i].size(); j++) {
        // (#679) Just ignore empty program
        if (fused_circuits[i].size() == 0) {
          (*output_tensor)(i, j) = -2.0;
          continue;
        }
        float exp_v = 0.0;
        OP_REQUIRES_OK(context,
                       ComputeExpectationQsim(pauli_sums[i][j], sim, ss, sv,
                                              scratch, &exp_v));
        (*output_tensor)(i, j) = exp_v;
      }
    }
  }

  void ComputeSmall(
      const std::vector<int>& num_qubits, const int max_num_qubits,
      const std::vector<std::vector<qsim::GateFused<QsimGate>>>& fused_circuits,
      const std::vector<std::vector<PauliSum>>& pauli_sums,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<float, 1>::Matrix* output_tensor) {
    using Simulator = qsim::SimulatorCUDA<float>;
    using StateSpace = Simulator::StateSpace;
    // TFQ GPU
    Simulator::Parameter param_sim;
    param_sim.num_threads = num_threads_in_sim_;

    StateSpace::Parameter param_ss;
    param_ss.num_threads = thread_per_block_;
    param_ss.num_dblocks = block_count_;

    const int output_dim_op_size = output_tensor->dimension(1);

    Status compute_status = Status::OK();
    auto c_lock = tensorflow::mutex();
    auto DoWork = [&](int start, int end) {
      int old_batch_index = -2;
      int cur_batch_index = -1;
      int largest_nq = 1;
      int cur_op_index;

      // Begin simulation.
      // Think later, d_wf, d_idx, d_ms, d_xss);
      auto sim = Simulator(param_sim);
      auto ss = StateSpace(param_ss);
      auto sv = ss.Create(largest_nq);
      auto scratch = ss.Create(largest_nq);
      for (int i = start; i < end; i++) {
        cur_batch_index = i / output_dim_op_size;
        cur_op_index = i % output_dim_op_size;

        const int nq = num_qubits[cur_batch_index];

        // (#679) Just ignore empty program
        if (fused_circuits[cur_batch_index].size() == 0) {
          (*output_tensor)(cur_batch_index, cur_op_index) = -2.0;
          continue;
        }

        if (cur_batch_index != old_batch_index) {
          // We've run into a new state vector we must compute.
          // Only compute a new state vector when we have to.
          if (nq > largest_nq) {
            largest_nq = nq;
            sv = ss.Create(largest_nq);
            scratch = ss.Create(largest_nq);
          }
          // no need to update scratch_state since ComputeExpectation
          // will take care of things for us.
          ss.SetStateZero(sv);
          for (int j = 0; j < fused_circuits[cur_batch_index].size(); j++) {
            qsim::ApplyFusedGate(sim, fused_circuits[cur_batch_index][j], sv);
          }
        }

        float exp_v = 0.0;
        NESTED_FN_STATUS_SYNC(
            compute_status,
            ComputeExpectationQsim(pauli_sums[cur_batch_index][cur_op_index],
                                   sim, ss, sv, scratch, &exp_v),
            c_lock);
        (*output_tensor)(cur_batch_index, cur_op_index) = exp_v;
        old_batch_index = cur_batch_index;
      }
    };

    const int64_t num_cycles = static_cast<int64_t>(cpu_cycle_);
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        fused_circuits.size() * output_dim_op_size, num_cycles, DoWork);
    OP_REQUIRES_OK(context, compute_status);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqSimulateExpectationGpuCpu").Device(tensorflow::DEVICE_CPU),
    TfqSimulateExpectationOpGpuCpu);

REGISTER_OP("TfqSimulateExpectationGpuCpu")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("pauli_sums: string")
    .Output("expectations: float")
    .Attr("num_threads_in_sim: int >= 32 = 32")
    .Attr("block_count: int >= 2 = 2")
    .Attr("thread_per_block: int >= 32 = 32")
    .Attr("cpu_cycle: int >= 1 = 1")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      tensorflow::shape_inference::ShapeHandle pauli_sums_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &pauli_sums_shape));

      tensorflow::shape_inference::DimensionHandle output_rows =
          c->Dim(programs_shape, 0);
      tensorflow::shape_inference::DimensionHandle output_cols =
          c->Dim(pauli_sums_shape, 1);
      c->set_output(0, c->Matrix(output_rows, output_cols));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
