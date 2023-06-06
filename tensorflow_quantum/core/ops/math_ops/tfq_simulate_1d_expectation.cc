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
#include "../qsim/lib/formux.h"
#include "../qsim/lib/gate_appl.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/mps_simulator.h"
#include "../qsim/lib/mps_statespace.h"
#include "../qsim/lib/seqfor.h"
#include "../qsim/lib/simmux.h"
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
#include "tensorflow_quantum/core/src/program_resolution.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::tensorflow::Status;
using ::tfq::proto::PauliSum;
using ::tfq::proto::Program;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

class TfqSimulateMPS1DExpectationOp : public tensorflow::OpKernel {
 public:
  explicit TfqSimulateMPS1DExpectationOp(
      tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    // Get the bond dimension of MPS
    // Checked that bond_dim is a positive integer >= 2 by QSim definition.
    OP_REQUIRES_OK(context, context->GetAttr("bond_dim", &bond_dim_));
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
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_tensor = output->matrix<float>();

    // Parse program protos.
    std::vector<Program> programs;
    std::vector<int> num_qubits;
    std::vector<std::vector<PauliSum>> pauli_sums;

    // TODO: remove endianness workaround introduced here:
    // https://github.com/tensorflow/quantum/pull/610
    // once https://github.com/quantumlib/qsim/issues/492
    // is resolved.
    OP_REQUIRES_OK(context,
                   GetProgramsAndNumQubits(context, &programs, &num_qubits,
                                           &pauli_sums, true));

    std::vector<SymbolMap> maps;
    OP_REQUIRES_OK(context, GetSymbolMaps(context, &maps));

    OP_REQUIRES(context, programs.size() == maps.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of circuits and symbol_values do not match. Got ",
                    programs.size(), " circuits and ", maps.size(),
                    " symbol values.")));

    // Construct qsim circuits.
    std::vector<QsimCircuit> qsim_circuits(programs.size(), QsimCircuit());
    std::vector<QsimFusedCircuit> fused_circuits(programs.size(),
                                                 QsimFusedCircuit({}));
    Status parse_status = ::tensorflow::Status();
    auto p_lock = tensorflow::mutex();
    auto construct_f = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        Status local =
            QsimCircuitFromProgram(programs[i], maps[i], num_qubits[i],
                                   &qsim_circuits[i], &fused_circuits[i]);
        // If parsing works, check MPS constraints.
        if (local.ok()) {
          local = CheckMPSSupported(programs[i]);
        }
        NESTED_FN_STATUS_SYNC(parse_status, local, p_lock);
      }
    };

    const int num_cycles = 1000;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        output_dim_batch_size, num_cycles, construct_f);
    OP_REQUIRES_OK(context, parse_status);

    // Find largest circuit for tensor size padding and allocate
    // the output tensor.
    int max_num_qubits = 0;
    int min_num_qubits = 1 << 30;
    for (const int num : num_qubits) {
      max_num_qubits = std::max(max_num_qubits, num);
      min_num_qubits = std::min(min_num_qubits, num);
    }

    OP_REQUIRES(context, min_num_qubits > 3,
                tensorflow::errors::InvalidArgument(
                    "All input circuits require minimum 3 qubits."));

    // Since MPS simulations have much smaller memory footprint,
    // we do not need a ComputeLarge like we do for state vector simulation.
    ComputeSmall(num_qubits, max_num_qubits, qsim_circuits, pauli_sums, context,
                 &output_tensor);
  }

 private:
  int bond_dim_;

  void ComputeSmall(const std::vector<int>& num_qubits,
                    const int max_num_qubits,
                    const std::vector<QsimCircuit>& unfused_circuits,
                    const std::vector<std::vector<PauliSum>>& pauli_sums,
                    tensorflow::OpKernelContext* context,
                    tensorflow::TTypes<float>::Matrix* output_tensor) {
    using Simulator = qsim::mps::MPSSimulator<qsim::For, float>;
    using StateSpace = Simulator::MPSStateSpace_;

    const int output_dim_op_size = output_tensor->dimension(1);

    Status compute_status = ::tensorflow::Status();
    auto c_lock = tensorflow::mutex();
    auto DoWork = [&](int start, int end) {
      int old_batch_index = -2;
      int cur_batch_index = -1;
      int largest_nq = 1;
      int cur_op_index;

      // Note: ForArgs in MPSSimulator and MPSStateState are currently unused.
      // So, this 1 is a dummy for qsim::For.
      Simulator sim = Simulator(1);
      StateSpace ss = StateSpace(1);
      auto sv = ss.Create(largest_nq, bond_dim_);
      auto scratch = ss.Create(largest_nq, bond_dim_);
      for (int i = start; i < end; i++) {
        cur_batch_index = i / output_dim_op_size;
        cur_op_index = i % output_dim_op_size;

        const int nq = num_qubits[cur_batch_index];

        // (#679) Just ignore empty program
        auto unfused_gates = unfused_circuits[cur_batch_index].gates;
        if (unfused_gates.size() == 0) {
          (*output_tensor)(cur_batch_index, cur_op_index) = -2.0;
          continue;
        }

        if (cur_batch_index != old_batch_index) {
          // We've run into a new state vector we must compute.
          // Only compute a new state vector when we have to.
          if (nq > largest_nq) {
            largest_nq = nq;
            sv = ss.Create(largest_nq, bond_dim_);
            scratch = ss.Create(largest_nq, bond_dim_);
          }
          // no need to update scratch_state since ComputeExpectationMPSQsim
          // will take care of things for us.
          ss.SetStateZero(sv);
          for (auto gate : unfused_gates) {
            // Can't fuse, since this might break nearest neighbor constraints.
            qsim::ApplyGate(sim, gate, sv);
          }
        }

        // Compute expectation values without fusing gates.
        float exp_v = 0.0;
        NESTED_FN_STATUS_SYNC(
            compute_status,
            ComputeExpectationQsim(pauli_sums[cur_batch_index][cur_op_index],
                                   sim, ss, sv, scratch, &exp_v, false),
            c_lock);
        (*output_tensor)(cur_batch_index, cur_op_index) = exp_v;
        old_batch_index = cur_batch_index;
      }
    };

    const int64_t num_cycles =
        200 * (int64_t(1) << static_cast<int64_t>(max_num_qubits));
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        unfused_circuits.size() * output_dim_op_size, num_cycles, DoWork);
    OP_REQUIRES_OK(context, compute_status);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqSimulateMPS1DExpectation").Device(tensorflow::DEVICE_CPU),
    TfqSimulateMPS1DExpectationOp);

REGISTER_OP("TfqSimulateMPS1DExpectation")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("pauli_sums: string")
    .Output("expectations: float")
    .Attr("bond_dim: int >= 4 = 4")
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

      return ::tensorflow::Status();
    });

}  // namespace tfq
