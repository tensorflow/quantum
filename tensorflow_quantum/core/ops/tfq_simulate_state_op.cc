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

#include <string>

#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"
#include "tensorflow_quantum/core/qsim/mux.h"
#include "tensorflow_quantum/core/qsim/state_space.h"
#include "tensorflow_quantum/core/src/circuit_parser.h"
#include "tensorflow_quantum/core/src/program_resolution.h"

namespace tfq {

using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;
using ::tfq::Circuit;
using ::tfq::CircuitFromProgram;
using ::tfq::qsim::GetStateSpace;
using ::tfq::qsim::StateSpace;

class TfqSimulateStateOp : public tensorflow::OpKernel {
 public:
  explicit TfqSimulateStateOp(tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override {
    // TODO (mbbrough): add more dimension checks for other inputs here.
    DCHECK_EQ(3, context->num_inputs());

    std::vector<Program> programs;
    std::vector<int> num_qubits;
    OP_REQUIRES_OK(context,
                   GetProgramsAndNumQubits(context, &programs, &num_qubits));
    std::vector<SymbolMap> maps;
    OP_REQUIRES_OK(context, GetSymbolMaps(context, &maps));

    OP_REQUIRES(
        context, maps.size() == programs.size(),
        tensorflow::errors::InvalidArgument(absl::StrCat(
            "Number of circuits and values do not match. Got ", programs.size(),
            " circuits and ", maps.size(), " values.")));

    int max_num_qubits = 0;
    for (const int num : num_qubits) {
      max_num_qubits = std::max(max_num_qubits, num);
    }

    // TODO(pmassey): Investigate creating a matrix that isn't just the maximum
    // required size.
    const int output_dim_size = maps.size();
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(output_dim_size);
    output_shape.AddDim(1 << max_num_qubits);

    tensorflow::Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_tensor = output->matrix<std::complex<float>>();

    auto DoWork = [&](int start, int end) {
      std::unique_ptr<StateSpace> state =
          std::unique_ptr<StateSpace>(GetStateSpace(1, 1));
      int old_num_qubits = -1;
      for (int i = start; i < end; i++) {
        Program program = programs[i];
        const int num = num_qubits[i];
        OP_REQUIRES_OK(context, ResolveSymbols(maps[i], &program));

        // QSim work below
        Circuit circuit;
        OP_REQUIRES_OK(context, CircuitFromProgram(program, num, &circuit));

        // TODO(mbbrough): Update this allocation hack so that a StateSpace
        //  object can grow it's memory dynamically to larger and larger size
        //  without ever having to call free (until the very end). This is
        //  tricky to implement because right now certain statespaces can't
        //  simulate all states and we use StateSpaceSlow for smaller circuits.
        if (num != old_num_qubits) {
          state.reset(GetStateSpace(num, 1));
          state->CreateState();
        }
        state->SetStateZero();
        OP_REQUIRES_OK(context, state->Update(circuit));
        uint64_t state_size = state->GetDimension();
        for (uint64_t j = 0; j < state_size; j++) {
          output_tensor(i, j) = state->GetAmpl(j);
        }
        for (uint64_t j = state_size; j < (uint64_t(1) << max_num_qubits);
             j++) {
          output_tensor(i, j) = std::complex<float>(-2, 0);
        }
        old_num_qubits = num;
      }
    };

    const int block_size = GetBlockSize(context, output_dim_size);
    context->device()
        ->tensorflow_cpu_worker_threads()
        ->workers->TransformRangeConcurrently(block_size, output_dim_size,
                                              DoWork);
  }
};

REGISTER_KERNEL_BUILDER(Name("TfqSimulateState").Device(tensorflow::DEVICE_CPU),
                        TfqSimulateStateOp);

REGISTER_OP("TfqSimulateState")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Output("wavefunction: complex64")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      c->set_output(
          0, c->MakeShape(
                 {c->Dim(programs_shape, 0),
                  tensorflow::shape_inference::InferenceContext::kUnknownDim}));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
