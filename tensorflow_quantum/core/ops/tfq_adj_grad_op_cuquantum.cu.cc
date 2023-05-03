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

#include <custatevec.h>

#include <memory>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/seqfor.h"
#include "../qsim/lib/simmux_gpu.h"
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
#include "tensorflow_quantum/core/src/adj_util.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

namespace {
// TODO(jaeyoo): Temorary hack for BulkSetAmpl with cuda ops.
// Updates qsim custatevec side BulkSetAmple ops, and remove these utilities.
template <typename FP, unsigned warp_size = 32>
__global__ void BulkSetAmplKernel(uint64_t mask, uint64_t bits, FP re, FP im,
                                  bool exclude, FP* state) {
  uint64_t k1 = uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  uint64_t k2 = 2 * k1 - threadIdx.x % warp_size;

  bool set = ((k1 & mask) == bits) ^ exclude;

  if (set) {
    state[k2] = re;
    state[k2 + warp_size] = im;
  }
}

// Sets state[i] = complex(re, im) where (i & mask) == bits.
// if `exclude` is true then the criteria becomes (i & mask) != bits.
template <typename fp_type>
void BulkSetAmpl(qsim::SimulatorCuStateVec<float>::StateSpace::State& state,
                 uint64_t mask, uint64_t bits, fp_type re, fp_type im,
                 bool exclude = false) {
  uint64_t size = uint64_t{1} << state.num_qubits();

  unsigned threads = std::min(size, uint64_t{512});
  unsigned blocks = size / threads;

  BulkSetAmplKernel<<<blocks, threads>>>(mask, bits, re, im, exclude,
                                         state.get());
  cudaPeekAtLastError();
  cudaDeviceSynchronize();
}
}  // namespace

using ::tensorflow::Status;
using ::tfq::proto::PauliSum;
using ::tfq::proto::Program;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

class TfqAdjointGradientCuquantumOp : public tensorflow::OpKernel {
 public:
  explicit TfqAdjointGradientCuquantumOp(
      tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    // create handles for simulator
    cublasCreate(&cublas_handle_);
    custatevecCreate(&custatevec_handle_);
  }

  ~TfqAdjointGradientCuquantumOp() {
    // destroy handles in sync with simulator lifetime
    cublasDestroy(cublas_handle_);
    custatevecDestroy(custatevec_handle_);
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    // TODO (mbbrough): add more dimension checks for other inputs here.
    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 5,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 5 inputs, got ", num_inputs, " inputs.")));

    // Create the output Tensor.
    const int output_dim_batch_size = context->input(0).dim_size(0);
    const int output_dim_param_size = context->input(2).dim_size(1);
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(output_dim_batch_size);
    output_shape.AddDim(output_dim_param_size);

    tensorflow::Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
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
    std::vector<std::vector<qsim::GateFused<QsimGate>>> full_fuse(
        programs.size(), std::vector<qsim::GateFused<QsimGate>>({}));
    std::vector<std::vector<std::vector<qsim::GateFused<QsimGate>>>>
        partial_fused_circuits(
            programs.size(),
            std::vector<std::vector<qsim::GateFused<QsimGate>>>({}));

    // track metadata.
    std::vector<std::vector<tfq::GateMetaData>> gate_meta(
        programs.size(), std::vector<tfq::GateMetaData>({}));

    // track gradients
    std::vector<std::vector<GradientOfGate>> gradient_gates(
        programs.size(), std::vector<GradientOfGate>({}));

    Status parse_status = ::tensorflow::Status();
    auto p_lock = tensorflow::mutex();
    auto construct_f = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        Status local = QsimCircuitFromProgram(programs[i], maps[i],
                                              num_qubits[i], &qsim_circuits[i],
                                              &full_fuse[i], &gate_meta[i]);
        NESTED_FN_STATUS_SYNC(parse_status, local, p_lock);
        CreateGradientCircuit(qsim_circuits[i], gate_meta[i],
                              &partial_fused_circuits[i], &gradient_gates[i]);
      }
    };

    const int num_cycles = 1000;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        programs.size(), num_cycles, construct_f);
    OP_REQUIRES_OK(context, parse_status);

    // Get downstream gradients.
    std::vector<std::vector<float>> downstream_grads;
    OP_REQUIRES_OK(context, GetPrevGrads(context, &downstream_grads));

    OP_REQUIRES(context, downstream_grads.size() == programs.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of gradients and circuits do not match. Got ",
                    downstream_grads.size(), " gradients and ", programs.size(),
                    " circuits.")));

    OP_REQUIRES(
        context, context->input(4).dim_size(1) == context->input(3).dim_size(1),
        tensorflow::errors::InvalidArgument(absl::StrCat(
            "Number of gradients and pauli sum dimension do not match. Got ",
            context->input(4).dim_size(1), " gradient entries and ",
            context->input(3).dim_size(1), " paulis per circuit.")));

    int max_num_qubits = 0;
    for (const int num : num_qubits) {
      max_num_qubits = std::max(max_num_qubits, num);
    }

    output_tensor.setZero();

    ComputeLarge(num_qubits, qsim_circuits, maps, full_fuse,
                 partial_fused_circuits, pauli_sums, gradient_gates,
                 downstream_grads, context, &output_tensor);
  }

 private:
  cublasHandle_t cublas_handle_;
  custatevecHandle_t custatevec_handle_;

  void ComputeLarge(
      const std::vector<int>& num_qubits,
      const std::vector<QsimCircuit>& qsim_circuits,
      const std::vector<SymbolMap>& maps,
      const std::vector<std::vector<qsim::GateFused<QsimGate>>>& full_fuse,
      const std::vector<std::vector<std::vector<qsim::GateFused<QsimGate>>>>&
          partial_fused_circuits,
      const std::vector<std::vector<PauliSum>>& pauli_sums,
      const std::vector<std::vector<tfq::GradientOfGate>>& gradient_gates,
      const std::vector<std::vector<float>>& downstream_grads,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<float, 1>::Matrix* output_tensor) {
    // Instantiate qsim objects.
    using Simulator = qsim::SimulatorCuStateVec<float>;
    using StateSpace = Simulator::StateSpace;

    // Begin simulation.
    int largest_nq = 1;
    Simulator sim = Simulator(cublas_handle_, custatevec_handle_);
    StateSpace ss = StateSpace(cublas_handle_, custatevec_handle_);
    auto sv = ss.Create(largest_nq);
    auto scratch = ss.Create(largest_nq);
    auto scratch2 = ss.Create(largest_nq);

    for (size_t i = 0; i < partial_fused_circuits.size(); i++) {
      int nq = num_qubits[i];

      if (nq > largest_nq) {
        // need to switch to larger statespace.
        largest_nq = nq;
        sv = ss.Create(largest_nq);
        scratch = ss.Create(largest_nq);
        scratch2 = ss.Create(largest_nq);
      }

      // (#679) Just ignore empty program
      if (qsim_circuits[i].gates.size() == 0) {
        continue;
      }

      ss.SetStateZero(sv);
      for (size_t j = 0; j < full_fuse[i].size(); j++) {
        qsim::ApplyFusedGate(sim, full_fuse[i][j], sv);
      }

      // sv now contains psi
      // scratch contains (sum_j paulis_sums[i][j] * downstream_grads[j])|psi>
      // scratch2 now contains psi as well.
      [[maybe_unused]] Status unused = AccumulateOperators(pauli_sums[i], downstream_grads[i],
                                           sim, ss, sv, scratch2, scratch);

      for (int j = partial_fused_circuits[i].size() - 1; j >= 0; j--) {
        for (int k = partial_fused_circuits[i][j].size() - 1; k >= 0; k--) {
          ApplyFusedGateDagger(sim, partial_fused_circuits[i][j][k], sv);
          ApplyFusedGateDagger(sim, partial_fused_circuits[i][j][k], scratch);
        }
        if (j == 0) {
          // last layer will have no parametrized gates so can break.
          break;
        }

        // Hit a parameterized gate.
        // todo fix this copy.
        auto cur_gate = qsim_circuits[i].gates[gradient_gates[i][j - 1].index];
        ApplyGateDagger(sim, cur_gate, sv);

        // if applicable compute control qubit mask and control value bits.
        uint64_t mask = 0;
        uint64_t cbits = 0;
        for (size_t k = 0; k < cur_gate.controlled_by.size(); k++) {
          uint64_t control_loc = cur_gate.controlled_by[k];
          mask |= uint64_t{1} << control_loc;
          cbits |= ((cur_gate.cmask >> k) & 1) << control_loc;
        }

        for (size_t k = 0; k < gradient_gates[i][j - 1].grad_gates.size();
             k++) {
          // Copy sv onto scratch2 in anticipation of non-unitary "gradient
          // gate".
          ss.Copy(sv, scratch2);
          if (!cur_gate.controlled_by.empty()) {
            // Gradient of controlled gates puts zeros on diagonal which is
            // the same as collapsing the state and then applying the
            // non-controlled version of the gradient gate.
            BulkSetAmpl<float>(scratch2, mask, cbits, 0, 0, true);
          }
          qsim::ApplyGate(sim, gradient_gates[i][j - 1].grad_gates[k],
                          scratch2);

          // don't need not-found check since this is done upstream already.
          const auto it = maps[i].find(gradient_gates[i][j - 1].params[k]);
          const int loc = it->second.first;
          // Apply finite differencing for adjoint gradients.
          // Finite differencing enables applying multiple `gradient_gate`
          // of a symbol at the same circuit. For analytic methods like
          // parameter-shift we need to apply a single `gradient_gate`
          // per a symbol.
          (*output_tensor)(i, loc) += ss.RealInnerProduct(scratch2, scratch) +
                                      ss.RealInnerProduct(scratch, scratch2);
        }
        ApplyGateDagger(sim, cur_gate, scratch);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqAdjointGradientCuquantum").Device(tensorflow::DEVICE_CPU),
    TfqAdjointGradientCuquantumOp);

REGISTER_OP("TfqAdjointGradientCuquantum")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("pauli_sums: string")
    .Input("downstream_grads: float")
    .Output("grads: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      tensorflow::shape_inference::ShapeHandle pauli_sums_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &pauli_sums_shape));

      tensorflow::shape_inference::ShapeHandle downstream_grads_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &downstream_grads_shape));

      tensorflow::shape_inference::DimensionHandle output_rows =
          c->Dim(programs_shape, 0);
      tensorflow::shape_inference::DimensionHandle output_cols =
          c->Dim(symbol_names_shape, 0);
      c->set_output(0, c->Matrix(output_rows, output_cols));

      return ::tensorflow::Status();
    });

}  // namespace tfq
