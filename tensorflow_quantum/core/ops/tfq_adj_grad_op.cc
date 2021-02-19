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

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/seqfor.h"
#include "../qsim/lib/simmux.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/src/adj_util.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;
using ::tfq::proto::PauliSum;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

class TfqAdjointGradientOp : public tensorflow::OpKernel {
 public:
  explicit TfqAdjointGradientOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

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

    auto construct_f = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        OP_REQUIRES_OK(
            context, QsimCircuitFromProgram(programs[i], maps[i], num_qubits[i],
                                            &qsim_circuits[i], &full_fuse[i],
                                            &gate_meta[i]));
        CreateGradientCircuit(qsim_circuits[i], gate_meta[i],
                              &partial_fused_circuits[i], &gradient_gates[i]);
      }
    };

    const int num_cycles = 1000;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        programs.size(), num_cycles, construct_f);

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

    // Cross reference with standard google cloud compute instances
    // Memory ~= 2 * num_threads * (2 * 64 * 2 ** num_qubits in circuits)
    // e2s2 = 2 CPU, 8GB -> Can safely do 25 since Memory = 4GB
    // e2s4 = 4 CPU, 16GB -> Can safely do 25 since Memory = 8GB
    // ...
    // This method creates 3 big state vectors per thread so reducing size
    // here slightly.
    if (max_num_qubits >= 25 || programs.size() == 1) {
      ComputeLarge(num_qubits, qsim_circuits, maps, full_fuse,
                   partial_fused_circuits, pauli_sums, gradient_gates,
                   downstream_grads, context, &output_tensor);
    } else {
      ComputeSmall(num_qubits, max_num_qubits, qsim_circuits, maps, full_fuse,
                   partial_fused_circuits, pauli_sums, gradient_gates,
                   downstream_grads, context, &output_tensor);
    }
  }

 private:
  void ComputeSmall(
      const std::vector<int>& num_qubits, const int max_num_qubits,
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
    const auto tfq_for = qsim::SequentialFor(1);
    using Simulator = qsim::Simulator<const qsim::SequentialFor&>;
    using StateSpace = Simulator::StateSpace;

    auto DoWork = [&](int start, int end) {
      // Begin simulation.
      int largest_nq = 1;
      Simulator sim = Simulator(tfq_for);
      StateSpace ss = StateSpace(tfq_for);
      auto sv = ss.Create(largest_nq);
      auto scratch = ss.Create(largest_nq);
      auto scratch2 = ss.Create(largest_nq);

      for (int i = start; i < end; i++) {
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
        for (int j = 0; j < full_fuse[i].size(); j++) {
          qsim::ApplyFusedGate(sim, full_fuse[i][j], sv);
        }

        // sv now contains psi
        // scratch contains (sum_j paulis_sums[i][j] * downstream_grads[j])|psi>
        // scratch2 now contains psi as well.
        AccumulateOperators(pauli_sums[i], downstream_grads[i], sim, ss, sv,
                            scratch2, scratch);

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
          auto cur_gate =
              qsim_circuits[i].gates[gradient_gates[i][j - 1].index];

          ApplyGateDagger(sim, cur_gate, sv);

          // if applicable compute control qubit mask and control value bits.
          uint64_t mask = 0;
          uint64_t cbits = 0;
          for (int k = 0; k < cur_gate.controlled_by.size(); k++) {
            uint64_t control_loc = cur_gate.controlled_by[k];
            mask |= uint64_t{1} << control_loc;
            cbits |= ((cur_gate.cmask >> k) & 1) << control_loc;
          }

          for (int k = 0; k < gradient_gates[i][j - 1].grad_gates.size(); k++) {
            // Copy sv onto scratch2 in anticipation of non-unitary "gradient
            // gate".
            ss.Copy(sv, scratch2);
            if (!cur_gate.controlled_by.empty()) {
              // Gradient of controlled gattes puts zeros on diagonal which is
              // the same as collapsing the state and then applying the
              // non-controlled version of the gradient gate.
              ss.BulkSetAmpl(scratch2, mask, cbits, 0, 0, true);
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
    };

    const int64_t num_cycles =
        200 * (int64_t(1) << static_cast<int64_t>(max_num_qubits));
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        qsim_circuits.size(), num_cycles, DoWork);
  }

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
    const auto tfq_for = tfq::QsimFor(context);
    using Simulator = qsim::Simulator<const tfq::QsimFor&>;
    using StateSpace = Simulator::StateSpace;

    // Begin simulation.
    int largest_nq = 1;
    Simulator sim = Simulator(tfq_for);
    StateSpace ss = StateSpace(tfq_for);
    auto sv = ss.Create(largest_nq);
    auto scratch = ss.Create(largest_nq);
    auto scratch2 = ss.Create(largest_nq);

    for (int i = 0; i < partial_fused_circuits.size(); i++) {
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
      for (int j = 0; j < full_fuse[i].size(); j++) {
        qsim::ApplyFusedGate(sim, full_fuse[i][j], sv);
      }

      // sv now contains psi
      // scratch contains (sum_j paulis_sums[i][j] * downstream_grads[j])|psi>
      // scratch2 now contains psi as well.
      AccumulateOperators(pauli_sums[i], downstream_grads[i], sim, ss, sv,
                          scratch2, scratch);

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
        for (int k = 0; k < cur_gate.controlled_by.size(); k++) {
          uint64_t control_loc = cur_gate.controlled_by[k];
          mask |= uint64_t{1} << control_loc;
          cbits |= ((cur_gate.cmask >> k) & 1) << control_loc;
        }

        for (int k = 0; k < gradient_gates[i][j - 1].grad_gates.size(); k++) {
          // Copy sv onto scratch2 in anticipation of non-unitary "gradient
          // gate".
          ss.Copy(sv, scratch2);
          if (!cur_gate.controlled_by.empty()) {
            // Gradient of controlled gattes puts zeros on diagonal which is
            // the same as collapsing the state and then applying the
            // non-controlled version of the gradient gate.
            ss.BulkSetAmpl(scratch2, mask, cbits, 0, 0, true);
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
    Name("TfqAdjointGradient").Device(tensorflow::DEVICE_CPU),
    TfqAdjointGradientOp);

REGISTER_OP("TfqAdjointGradient")
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

      return tensorflow::Status::OK();
    });

}  // namespace tfq
