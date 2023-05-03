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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"
#include "tensorflow_quantum/core/proto/program.pb.h"

namespace tfq {

using ::tfq::proto::Moment;
using ::tfq::proto::Program;

class TfqCircuitAppendOp : public tensorflow::OpKernel {
 public:
  explicit TfqCircuitAppendOp(tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override {
    std::vector<Program> programs;
    std::vector<Program> programs_to_append;

    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 2,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 2 inputs, got ", num_inputs, " inputs.")));

    OP_REQUIRES_OK(context, GetProgramsAndProgramsToAppend(
                                context, &programs, &programs_to_append));

    tensorflow::Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, context->input(0).shape(), &output));
    auto output_tensor = output->flat<tensorflow::tstring>();

    auto DoWork = [&](int start, int end) {
      std::string temp;
      for (int i = start; i < end; i++) {
        for (size_t j = 0;
             j < programs_to_append.at(i).circuit().moments().size(); j++) {
          Moment *new_moment = programs.at(i).mutable_circuit()->add_moments();
          *new_moment = programs_to_append.at(i).circuit().moments(j);
        }
        programs.at(i).SerializeToString(&temp);
        output_tensor(i) = temp;
      }
    };

    const int output_dim_size = programs.size();
    const int block_size = GetBlockSize(context, output_dim_size);
    context->device()
        ->tensorflow_cpu_worker_threads()
        ->workers->TransformRangeConcurrently(block_size, output_dim_size,
                                              DoWork);
  }
};

REGISTER_KERNEL_BUILDER(Name("TfqAppendCircuit").Device(tensorflow::DEVICE_CPU),
                        TfqCircuitAppendOp);

REGISTER_OP("TfqAppendCircuit")
    .Input("programs: string")
    .Input("programs_to_append: string")
    .Output("programs_extended: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle programs_to_append_shape;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(1), 1, &programs_to_append_shape));

      c->set_output(0, c->input(0));

      return ::tensorflow::Status();
    });

}  // namespace tfq
