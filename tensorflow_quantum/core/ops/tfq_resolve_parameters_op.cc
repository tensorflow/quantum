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

#include "cirq_google/api/v2/program.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"
#include "tensorflow_quantum/core/src/program_resolution.h"

namespace tfq {

using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;

class TfqResolveParametersOp : public tensorflow::OpKernel {
 public:
  explicit TfqResolveParametersOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 3,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 3 inputs, got ", num_inputs, " inputs.")));

    tensorflow::Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, context->input(0).shape(), &output));
    auto output_tensor = output->flat<tensorflow::tstring>();

    std::vector<Program> programs;
    OP_REQUIRES_OK(context, ParsePrograms(context, "programs", &programs));
    std::vector<SymbolMap> maps;
    OP_REQUIRES_OK(context, GetSymbolMaps(context, &maps));

    OP_REQUIRES(
        context, maps.size() == programs.size(),
        tensorflow::errors::InvalidArgument(absl::StrCat(
            "Number of circuits and values do not match. Got ", programs.size(),
            " circuits and ", maps.size(), " values.")));

    Status parse_status = Status::OK();
    auto p_lock = tensorflow::mutex();
    auto DoWork = [&](int start, int end) {
      std::string temp;
      for (int i = start; i < end; i++) {
        Program program = programs[i];
        Status local = ResolveSymbols(maps[i], &program, false);
        NESTED_FN_STATUS_SYNC(parse_status, local, p_lock);
        program.SerializeToString(&temp);
        output_tensor(i) = temp;
      }
    };

    const int num_cycles = 1000;
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        programs.size(), num_cycles, DoWork);
    OP_REQUIRES_OK(context, parse_status);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqResolveParameters").Device(tensorflow::DEVICE_CPU),
    TfqResolveParametersOp);

REGISTER_OP("TfqResolveParameters")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Output("resolved_programs: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      c->set_output(0, c->input(0));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
