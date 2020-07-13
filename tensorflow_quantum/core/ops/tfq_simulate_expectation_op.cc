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
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;
using ::tfq::proto::PauliSum;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

class TfqSimulateExpectationOp : public tensorflow::OpKernel {
 public:
  explicit TfqSimulateExpectationOp(tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override {
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

    tensorflow::Tensor *output = nullptr;
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

    OP_REQUIRES(context, pauli_sums.size() == programs.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of circuits and PauliSums do not match. Got ",
                    programs.size(), " circuits and ", pauli_sums.size(),
                    " paulisums.")));

    // Construct qsim circuits.
    std::vector<QsimCircuit> qsim_circuits(programs.size(), QsimCircuit());
    std::vector<std::vector<qsim::GateFused<QsimGate>>> fused_circuits(
        programs.size(), std::vector<qsim::GateFused<QsimGate>>({}));

    auto construct_f = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        OP_REQUIRES_OK(context, QsimCircuitFromProgram(
                                    programs[i], maps[i], num_qubits[i],
                                    &qsim_circuits[i], &fused_circuits[i]));
      }
    };

    const int num_cycles = 1000;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        programs.size(), num_cycles, construct_f);

    // Instantiate qsim objects.
    const auto tfq_for = tfq::QsimFor(context);
    using Simulator = qsim::Simulator<const tfq::QsimFor &>;
    using StateSpace = Simulator::StateSpace;
    using State = StateSpace::State;

    // Begin simulation.
    int largest_nq = 1;
    State sv = StateSpace(largest_nq, tfq_for).CreateState();
    State scratch = StateSpace(largest_nq, tfq_for).CreateState();

    // Simulate programs one by one. Parallelizing over wavefunctions
    // we no longer parallelize over circuits. Each time we encounter a
    // a larger circuit we will grow the Statevector as nescessary.
    for (int i = 0; i < programs.size(); i++) {
      int nq = num_qubits[i];
      Simulator sim = Simulator(nq, tfq_for);
      StateSpace ss = StateSpace(nq, tfq_for);
      if (nq > largest_nq) {
        // need to switch to larger statespace.
        largest_nq = nq;
        sv = ss.CreateState();
        scratch = ss.CreateState();
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
        if (programs[i].circuit().moments().empty()) {
          output_tensor(i, j) = -2.0;
          continue;
        }
        float exp_v = 0.0;
        OP_REQUIRES_OK(context,
                       ComputeExpectationQsim(pauli_sums[i][j], sim, ss, sv,
                                              scratch, &exp_v));
        output_tensor(i, j) = exp_v;
      }
    }
    // just to be on the safe side.
    sv.release();
    scratch.release();
    qsim_circuits.clear();
    fused_circuits.clear();
    num_qubits.clear();
    maps.clear();
    pauli_sums.clear();
    programs.clear();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqSimulateExpectation").Device(tensorflow::DEVICE_CPU),
    TfqSimulateExpectationOp);

REGISTER_OP("TfqSimulateExpectation")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("pauli_sums: string")
    .Output("expectations: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
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
