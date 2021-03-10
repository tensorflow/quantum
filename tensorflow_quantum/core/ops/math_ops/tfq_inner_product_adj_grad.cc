/* Copyright 2021 The TensorFlow Quantum Authors. All Rights Reserved.

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
#include "tensorflow_quantum/core/src/adj_util.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;
using ::tfq::proto::PauliSum;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;
typedef std::vector<qsim::GateFused<QsimGate>> QsimFusedCircuit;

class TfqInnerProductAdjGradOp : public tensorflow::OpKernel {
 public:
  explicit TfqInnerProductAdjGradOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // TODO (mbbrough): add more dimension checks for other inputs here.
    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 5,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 5 inputs, got ", num_inputs, " inputs.")));

    // Create the output Tensor.
    const int output_dim_batch_size = context->input(0).dim_size(0);
    const int output_dim_internal_size = context->input(3).dim_size(1);
    const int output_dim_symbol_size = context->input(1).dim_size(0);
    OP_REQUIRES(context, output_dim_symbol_size > 0,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "The number of symbols must be a positive integer, got ",
                    output_dim_symbol_size, " symbols.")));
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(output_dim_batch_size);
    output_shape.AddDim(output_dim_symbol_size);

    tensorflow::Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_tensor = output->matrix<std::complex<float>>();

    // Parse program protos.
    std::vector<Program> programs;
    std::vector<int> num_qubits;
    std::vector<std::vector<Program>> other_programs;
    OP_REQUIRES_OK(context,
                   GetProgramsAndNumQubits(context, &programs, &num_qubits,
                                           &other_programs));

    std::vector<SymbolMap> maps;
    OP_REQUIRES_OK(context, GetSymbolMaps(context, &maps));

    OP_REQUIRES(context, programs.size() == maps.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of circuits and symbol_values do not match. Got ",
                    programs.size(), " circuits and ", maps.size(),
                    " symbol values.")));
    OP_REQUIRES(context, output_dim_symbol_size == maps[0].size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of symbols and symbol maps do not match. Got ",
                    output_dim_symbol_size, " symbols and ", maps[0].size(),
                    " symbol values.")));

    // Construct qsim circuits for programs.
    std::vector<QsimCircuit> qsim_circuits(programs.size(), QsimCircuit());
    std::vector<QsimFusedCircuit> fused_circuits(programs.size(),
                                                 QsimFusedCircuit({}));

    // track metadata.
    std::vector<std::vector<tfq::GateMetaData>> gate_meta(
        programs.size(), std::vector<tfq::GateMetaData>({}));

    // Construct qsim circuits.
    std::vector<std::vector<std::vector<qsim::GateFused<QsimGate>>>>
        partial_fused_circuits(
            programs.size(),
            std::vector<std::vector<qsim::GateFused<QsimGate>>>({}));

    // track gradients
    std::vector<std::vector<GradientOfGate>> gradient_gates(
        programs.size(), std::vector<GradientOfGate>({}));

    auto construct_f = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        OP_REQUIRES_OK(
            context, QsimCircuitFromProgram(programs[i], maps[i], num_qubits[i],
                                            &qsim_circuits[i],
                                            &fused_circuits[i], &gate_meta[i]));

        CreateGradientCircuit(qsim_circuits[i], gate_meta[i],
                              &partial_fused_circuits[i], &gradient_gates[i]);
      }
    };

    const int num_cycles = 1000;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        output_dim_batch_size, num_cycles, construct_f);

    // Construct qsim circuits for other_programs.
    std::vector<std::vector<QsimCircuit>> other_qsim_circuits(
        output_dim_batch_size,
        std::vector<QsimCircuit>(output_dim_internal_size, QsimCircuit()));
    std::vector<std::vector<QsimFusedCircuit>> other_fused_circuits(
        output_dim_batch_size,
        std::vector<QsimFusedCircuit>(output_dim_internal_size,
                                      QsimFusedCircuit({})));

    auto construct_f2 = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        int ii = i / output_dim_internal_size;
        int jj = i % output_dim_internal_size;
        Status status = QsimCircuitFromProgram(
            other_programs[ii][jj], {}, num_qubits[ii],
            &other_qsim_circuits[ii][jj], &other_fused_circuits[ii][jj]);
        OP_REQUIRES(context, status.ok(),
                    tensorflow::errors::InvalidArgument(absl::StrCat(
                        "Found symbols in other_programs.",
                        "No symbols are allowed in these circuits.")));
      }
    };

    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        output_dim_batch_size * output_dim_internal_size, num_cycles,
        construct_f2);

    int max_num_qubits = 0;
    for (const int num : num_qubits) {
      max_num_qubits = std::max(max_num_qubits, num);
    }

    // Get downstream gradients.
    std::vector<std::vector<float>> downstream_grads;
    OP_REQUIRES_OK(context, GetPrevGrads(context, &downstream_grads));

    OP_REQUIRES(context, downstream_grads.size() == programs.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of gradients and circuits do not match. Got ",
                    downstream_grads.size(), " gradients and ", programs.size(),
                    " circuits.")));

    OP_REQUIRES(context, downstream_grads[0].size() == output_dim_internal_size,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of gradients and other_programs do not match. Got ",
                    downstream_grads[0].size(), " gradient entries and ",
                    output_dim_internal_size, " other programs.")));

    output_tensor.setZero();

    // Cross reference with standard google cloud compute instances
    // Memory ~= 2 * num_threads * (2 * 64 * 2 ** num_qubits in circuits)
    // e2s2 = 2 CPU, 8GB -> Can safely do 23 since Memory = 4GB
    // e2s4 = 4 CPU, 16GB -> Can safely do 23 since Memory = 8GB
    // ...
    if (max_num_qubits >= 24 || output_dim_batch_size == 1) {
      ComputeLarge(num_qubits, maps, qsim_circuits, fused_circuits,
                   partial_fused_circuits, gradient_gates, other_fused_circuits,
                   downstream_grads, context, &output_tensor);
    } else {
      ComputeSmall(num_qubits, max_num_qubits, maps, qsim_circuits,
                   fused_circuits, partial_fused_circuits, gradient_gates,
                   other_fused_circuits, downstream_grads, context,
                   &output_tensor);
    }
  }

 private:
  void ComputeLarge(
      const std::vector<int>& num_qubits, const std::vector<SymbolMap>& maps,
      const std::vector<QsimCircuit>& qsim_circuits,
      const std::vector<QsimFusedCircuit>& fused_circuits,
      const std::vector<std::vector<std::vector<qsim::GateFused<QsimGate>>>>&
          partial_fused_circuits,
      const std::vector<std::vector<tfq::GradientOfGate>>& gradient_gates,
      const std::vector<std::vector<QsimFusedCircuit>>& other_fused_circuits,
      const std::vector<std::vector<float>>& downstream_grads,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<std::complex<float>>::Matrix* output_tensor) {
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

    // Simulate programs one by one. Parallelizing over state vectors
    // we no longer parallelize over circuits. Each time we encounter a
    // a larger circuit we will grow the Statevector as necessary.
    for (std::vector<std::vector<qsim::GateFused<QsimGate>>>::size_type i = 0;
         i < fused_circuits.size(); i++) {
      int nq = num_qubits[i];
      if (nq > largest_nq) {
        // need to switch to larger statespace.
        largest_nq = nq;
        sv = ss.Create(largest_nq);
        scratch = ss.Create(largest_nq);
        scratch2 = ss.Create(largest_nq);
      }
      ss.SetStateZero(sv);
      for (std::vector<qsim::GateFused<QsimGate>>::size_type j = 0;
           j < fused_circuits[i].size(); j++) {
        qsim::ApplyFusedGate(sim, fused_circuits[i][j], sv);
      }

      auto status = AccumulateFusedCircuits(
          downstream_grads, other_fused_circuits, sim, ss, scratch2, scratch);

      // now sv is |psi>
      // scratch contains sum_j downstream_grads[i][j]*|phi[i][j]>
      // Start adjoint differentiation.
      for (int l = partial_fused_circuits[i].size() - 1; l >= 0; l--) {
        for (int k = partial_fused_circuits[i][l].size() - 1; k >= 0; k--) {
          ApplyFusedGateDagger(sim, partial_fused_circuits[i][l][k], sv);
          ApplyFusedGateDagger(sim, partial_fused_circuits[i][l][k], scratch);
        }
        if (l == 0) {
          // last layer will have no parametrized gates so can break.
          break;
        }

        // Hit a parameterized gate.
        // todo fix this copy.
        auto cur_gate = qsim_circuits[i].gates[gradient_gates[i][l - 1].index];
        ApplyGateDagger(sim, cur_gate, sv);

        // if applicable compute control qubit mask and control value bits.
        uint64_t mask = 0;
        uint64_t cbits = 0;
        for (std::vector<unsigned int>::size_type k = 0;
             k < cur_gate.controlled_by.size(); k++) {
          uint64_t control_loc = cur_gate.controlled_by[k];
          mask |= uint64_t{1} << control_loc;
          cbits |= ((cur_gate.cmask >> k) & 1) << control_loc;
        }

        for (std::vector<QsimGate>::size_type k = 0;
             k < gradient_gates[i][l - 1].grad_gates.size(); k++) {
          // Copy sv onto scratch2 in anticipation of non-unitary "gradient
          // gate".
          ss.Copy(sv, scratch2);
          if (!cur_gate.controlled_by.empty()) {
            // Gradient of controlled gates puts zeros on diagonal which is
            // the same as collapsing the state and then applying the
            // non-controlled version of the gradient gate.
            ss.BulkSetAmpl(scratch2, mask, cbits, 0, 0, true);
          }
          qsim::ApplyGate(sim, gradient_gates[i][l - 1].grad_gates[k],
                          scratch2);

          // don't need not-found check since this is done upstream already.
          const auto it = maps[i].find(gradient_gates[i][l - 1].params[k]);
          const int loc = it->second.first;
          // Apply finite differencing for adjoint gradients.
          // Finite differencing enables applying multiple `gradient_gate`
          // of a symbol at the same circuit. For analytic methods like
          // parameter-shift we need to apply a single `gradient_gate`
          // per a symbol.
          std::complex<double> result = ss.InnerProduct(scratch2, scratch);
          (*output_tensor)(i, loc) +=
              std::complex<float>(static_cast<float>(result.real()),
                                  static_cast<float>(result.imag()));
        }
        ApplyGateDagger(sim, cur_gate, scratch);
      }
    }
  }

  void ComputeSmall(
      const std::vector<int>& num_qubits, const int max_num_qubits,
      const std::vector<SymbolMap>& maps,
      const std::vector<QsimCircuit>& qsim_circuits,
      const std::vector<QsimFusedCircuit>& fused_circuits,
      const std::vector<std::vector<std::vector<qsim::GateFused<QsimGate>>>>&
          partial_fused_circuits,
      const std::vector<std::vector<tfq::GradientOfGate>>& gradient_gates,
      const std::vector<std::vector<QsimFusedCircuit>>& other_fused_circuits,
      const std::vector<std::vector<float>>& downstream_grads,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<std::complex<float>>::Matrix* output_tensor) {
    const auto tfq_for = qsim::SequentialFor(1);
    using Simulator = qsim::Simulator<const qsim::SequentialFor&>;
    using StateSpace = Simulator::StateSpace;

    const int output_dim_internal_size = other_fused_circuits[0].size();

    auto DoWork = [&](int start, int end) {
      int old_batch_index = -2;
      int cur_batch_index = -1;
      int largest_nq = 1;
      int cur_internal_index;

      Simulator sim = Simulator(tfq_for);
      StateSpace ss = StateSpace(tfq_for);
      auto sv = ss.Create(largest_nq);
      auto sv_adj = ss.Create(largest_nq);
      auto scratch = ss.Create(largest_nq);
      auto scratch2 = ss.Create(largest_nq);
      for (int i = start; i < end; i++) {
        cur_batch_index = i / output_dim_internal_size;
        cur_internal_index = i % output_dim_internal_size;

        const int nq = num_qubits[cur_batch_index];

        if (cur_batch_index != old_batch_index) {
          // We've run into a new state vector we must compute.
          // Only compute a new state vector when we have to.
          if (nq > largest_nq) {
            largest_nq = nq;
            sv = ss.Create(largest_nq);
            sv_adj = ss.Create(largest_nq);
            scratch = ss.Create(largest_nq);
            scratch2 = ss.Create(largest_nq);
          }
          ss.SetStateZero(sv);
          for (std::vector<qsim::GateFused<QsimGate>>::size_type j = 0;
               j < fused_circuits[cur_batch_index].size(); j++) {
            qsim::ApplyFusedGate(sim, fused_circuits[cur_batch_index][j], sv);
          }
        }

        ss.SetStateZero(scratch);
        for (std::vector<qsim::GateFused<QsimGate>>::size_type k = 0;
             k <
             other_fused_circuits[cur_batch_index][cur_internal_index].size();
             k++) {
          qsim::ApplyFusedGate(
              sim, other_fused_circuits[cur_batch_index][cur_internal_index][k],
              scratch);
        }
        // now sv is |psi>, scratch is |phi>
        // Start adjoint differentiation.
        ss.Copy(sv, sv_adj);
        for (int l = partial_fused_circuits[cur_batch_index].size() - 1; l >= 0;
             l--) {
          for (int k = partial_fused_circuits[cur_batch_index][l].size() - 1;
               k >= 0; k--) {
            ApplyFusedGateDagger(
                sim, partial_fused_circuits[cur_batch_index][l][k], sv_adj);
            ApplyFusedGateDagger(
                sim, partial_fused_circuits[cur_batch_index][l][k], scratch);
          }
          if (l == 0) {
            // last layer will have no parametrized gates so can break.
            break;
          }

          // Hit a parameterized gate.
          // todo fix this copy.
          auto cur_gate =
              qsim_circuits[cur_batch_index]
                  .gates[gradient_gates[cur_batch_index][l - 1].index];
          ApplyGateDagger(sim, cur_gate, sv_adj);

          // if applicable compute control qubit mask and control value bits.
          uint64_t mask = 0;
          uint64_t cbits = 0;
          for (int k = 0; k < cur_gate.controlled_by.size(); k++) {
            uint64_t control_loc = cur_gate.controlled_by[k];
            mask |= uint64_t{1} << control_loc;
            cbits |= ((cur_gate.cmask >> k) & 1) << control_loc;
          }

          for (int k = 0;
               k < gradient_gates[cur_batch_index][l - 1].grad_gates.size();
               k++) {
            // Copy sv_adj onto scratch2 in anticipation of non-unitary
            // "gradient gate".
            ss.Copy(sv_adj, scratch2);
            if (!cur_gate.controlled_by.empty()) {
              // Gradient of controlled gates puts zeros on diagonal which is
              // the same as collapsing the state and then applying the
              // non-controlled version of the gradient gate.
              ss.BulkSetAmpl(scratch2, mask, cbits, 0, 0, true);
            }
            qsim::ApplyGate(
                sim, gradient_gates[cur_batch_index][l - 1].grad_gates[k],
                scratch2);

            // don't need not-found check since this is done upstream already.
            const auto it = maps[cur_batch_index].find(
                gradient_gates[cur_batch_index][l - 1].params[k]);
            const int loc = it->second.first;
            // Apply finite differencing for adjoint gradients.
            // Finite differencing enables applying multiple `gradient_gate`
            // of a symbol at the same circuit. For analytic methods like
            // parameter-shift we need to apply a single `gradient_gate`
            // per a symbol.
            std::complex<double> result = ss.InnerProduct(scratch2, scratch);
            (*output_tensor)(cur_batch_index, loc) +=
                (downstream_grads[cur_batch_index][cur_internal_index] *
                 std::complex<float>(static_cast<float>(result.real()),
                                     static_cast<float>(result.imag())));
          }
          ApplyGateDagger(sim, cur_gate, scratch);
        }
        old_batch_index = cur_batch_index;
      }
    };

    const int64_t num_cycles =
        200 * (int64_t(1) << static_cast<int64_t>(max_num_qubits));
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        fused_circuits.size() * output_dim_internal_size, num_cycles, DoWork);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqInnerProductAdjGrad").Device(tensorflow::DEVICE_CPU),
    TfqInnerProductAdjGradOp);

REGISTER_OP("TfqInnerProductAdjGrad")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("other_programs: string")
    .Input("downstream_grads: float")
    .Output("inner_products_adj_grad: complex64")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      tensorflow::shape_inference::ShapeHandle other_programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &other_programs_shape));

      tensorflow::shape_inference::ShapeHandle downstream_grads_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &downstream_grads_shape));

      tensorflow::shape_inference::DimensionHandle output_rows =
          c->Dim(programs_shape, 0);
      tensorflow::shape_inference::DimensionHandle output_cols =
          c->Dim(symbol_names_shape, 0);
      std::vector<tensorflow::shape_inference::DimensionHandle> dims = {
          output_rows, output_cols};
      c->set_output(0, c->MakeShape(dims));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
