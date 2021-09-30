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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/proto/program.pb.h"
#include "tensorflow_quantum/core/src/util_qsim.h"

namespace tfq {

using ::tensorflow::Status;
using ::tfq::proto::PauliSum;
using ::tfq::proto::Program;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;
typedef std::vector<qsim::GateFused<QsimGate>> QsimFusedCircuit;

class TfqInnerProductOp : public tensorflow::OpKernel {
 public:
  explicit TfqInnerProductOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // TODO (mbbrough): add more dimension checks for other inputs here.
    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 4,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 4 inputs, got ", num_inputs, " inputs.")));

    // Create the output Tensor.
    const int output_dim_batch_size = context->input(0).dim_size(0);
    const int output_dim_internal_size = context->input(3).dim_size(1);
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(output_dim_batch_size);
    output_shape.AddDim(output_dim_internal_size);

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

    // Construct qsim circuits for programs.
    std::vector<QsimCircuit> qsim_circuits(programs.size(), QsimCircuit());
    std::vector<QsimFusedCircuit> fused_circuits(programs.size(),
                                                 QsimFusedCircuit({}));

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
        output_dim_batch_size, num_cycles, construct_f);
    OP_REQUIRES_OK(context, parse_status);

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
        NESTED_FN_STATUS_SYNC(parse_status, status, p_lock);
      }
    };

    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        output_dim_batch_size * output_dim_internal_size, num_cycles,
        construct_f2);
    if (!parse_status.ok()) {
      OP_REQUIRES_OK(context,
                     tensorflow::errors::InvalidArgument(absl::StrCat(
                         "Found symbols in other_programs.",
                         "No symbols are allowed in these circuits.")));
    }

    int max_num_qubits = 0;
    for (const int num : num_qubits) {
      max_num_qubits = std::max(max_num_qubits, num);
    }

    // Cross reference with standard google cloud compute instances
    // Memory ~= 2 * num_threads * (2 * 64 * 2 ** num_qubits in circuits)
    // e2s2 = 2 CPU, 8GB -> Can safely do 25 since Memory = 4GB
    // e2s4 = 4 CPU, 16GB -> Can safely do 25 since Memory = 8GB
    // ...
    if (max_num_qubits >= 26 || output_dim_batch_size == 1) {
      ComputeLarge(num_qubits, fused_circuits, other_fused_circuits, context,
                   &output_tensor);
    } else {
      ComputeSmall(num_qubits, max_num_qubits, fused_circuits,
                   other_fused_circuits, context, &output_tensor);
    }
  }

 private:
  void ComputeLarge(
      const std::vector<int>& num_qubits,
      const std::vector<QsimFusedCircuit>& fused_circuits,
      const std::vector<std::vector<QsimFusedCircuit>>& other_fused_circuits,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<std::complex<float>, 1>::Matrix* output_tensor) {
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

    // Simulate programs one by one. Parallelizing over state vectors
    // we no longer parallelize over circuits. Each time we encounter a
    // a larger circuit we will grow the Statevector as necessary.
    for (std::vector<QsimFusedCircuit>::size_type i = 0;
         i < fused_circuits.size(); i++) {
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
      for (QsimFusedCircuit::size_type j = 0; j < fused_circuits[i].size();
           j++) {
        qsim::ApplyFusedGate(sim, fused_circuits[i][j], sv);
      }
      for (std::vector<QsimFusedCircuit>::size_type j = 0;
           j < other_fused_circuits[i].size(); j++) {
        // (#679) Just ignore empty program
        if (fused_circuits[i].size() == 0) {
          (*output_tensor)(i, j) = std::complex<float>(1, 0);
          continue;
        }

        ss.SetStateZero(scratch);
        for (QsimFusedCircuit k = 0; k < other_fused_circuits[i][j].size();
             k++) {
          qsim::ApplyFusedGate(sim, other_fused_circuits[i][j][k], scratch);
        }

        std::complex<double> result = ss.InnerProduct(sv, scratch);
        (*output_tensor)(i, j) =
            std::complex<float>(static_cast<float>(result.real()),
                                static_cast<float>(result.imag()));
      }
    }
  }

  void ComputeSmall(
      const std::vector<int>& num_qubits, const int max_num_qubits,
      const std::vector<QsimFusedCircuit>& fused_circuits,
      const std::vector<std::vector<QsimFusedCircuit>>& other_fused_circuits,
      tensorflow::OpKernelContext* context,
      tensorflow::TTypes<std::complex<float>, 1>::Matrix* output_tensor) {
    const auto tfq_for = qsim::SequentialFor(1);
    using Simulator = qsim::Simulator<const qsim::SequentialFor&>;
    using StateSpace = Simulator::StateSpace;

    const int output_dim_internal_size = output_tensor->dimension(1);

    auto DoWork = [&](int start, int end) {
      int old_batch_index = -2;
      int cur_batch_index = -1;
      int largest_nq = 1;
      int cur_internal_index;

      Simulator sim = Simulator(tfq_for);
      StateSpace ss = StateSpace(tfq_for);
      auto sv = ss.Create(largest_nq);
      auto scratch = ss.Create(largest_nq);
      for (int i = start; i < end; i++) {
        cur_batch_index = i / output_dim_internal_size;
        cur_internal_index = i % output_dim_internal_size;

        const int nq = num_qubits[cur_batch_index];

        // (#679) Just ignore empty program
        if (fused_circuits[cur_batch_index].size() == 0) {
          (*output_tensor)(cur_batch_index, cur_internal_index) =
              std::complex<float>(1, 0);
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
          for (QsimFusedCircuit::size_type j = 0;
               j < fused_circuits[cur_batch_index].size(); j++) {
            qsim::ApplyFusedGate(sim, fused_circuits[cur_batch_index][j], sv);
          }
        }

        ss.SetStateZero(scratch);
        for (QsimFusedCircuit::size_type k = 0;
             k <
             other_fused_circuits[cur_batch_index][cur_internal_index].size();
             k++) {
          qsim::ApplyFusedGate(
              sim, other_fused_circuits[cur_batch_index][cur_internal_index][k],
              scratch);
        }

        std::complex<double> result = ss.InnerProduct(sv, scratch);
        (*output_tensor)(cur_batch_index, cur_internal_index) =
            std::complex<float>(static_cast<float>(result.real()),
                                static_cast<float>(result.imag()));

        old_batch_index = cur_batch_index;
      }
    };

    const int64_t num_cycles =
        200 * (int64_t(1) << static_cast<int64_t>(max_num_qubits));
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        fused_circuits.size() * output_dim_internal_size, num_cycles, DoWork);
  }
};

REGISTER_KERNEL_BUILDER(Name("TfqInnerProduct").Device(tensorflow::DEVICE_CPU),
                        TfqInnerProductOp);

REGISTER_OP("TfqInnerProduct")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("other_programs: string")
    .Output("inner_products: complex64")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      tensorflow::shape_inference::ShapeHandle other_programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &other_programs_shape));

      tensorflow::shape_inference::DimensionHandle output_rows =
          c->Dim(programs_shape, 0);
      tensorflow::shape_inference::DimensionHandle output_cols =
          c->Dim(other_programs_shape, 1);
      c->set_output(0, c->Matrix(output_rows, output_cols));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
