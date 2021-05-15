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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"

namespace tfq {

using ::cirq::google::api::v2::Arg;
using ::cirq::google::api::v2::Moment;
using ::cirq::google::api::v2::Operation;
using ::cirq::google::api::v2::Program;

using ::tensorflow::Status;
using ::tensorflow::Tensor;

class TfqPsSymbolReplaceOp : public tensorflow::OpKernel {
 public:
  explicit TfqPsSymbolReplaceOp(tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override {
    std::vector<Program> programs;

    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 3,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 3 inputs, got ", num_inputs, " inputs.")));

    OP_REQUIRES_OK(context, ParsePrograms(context, "programs", &programs));

    // Parse the input string here.
    const Tensor *symbols_tensor;
    OP_REQUIRES_OK(context, context->input("symbols", &symbols_tensor));
    OP_REQUIRES(
        context, symbols_tensor->dims() == 1,
        tensorflow::errors::InvalidArgument(absl::StrCat(
            "symbols must be rank 1. Got rank ", symbols_tensor->dims(), ".")));

    const auto symbols = symbols_tensor->vec<tensorflow::tstring>();
    const size_t n_symbols = symbols.size();

    // Parse the replacement string here.
    const Tensor *replacement_symbols_tensor;
    OP_REQUIRES_OK(context, context->input("replacement_symbols",
                                           &replacement_symbols_tensor));
    OP_REQUIRES(context, replacement_symbols_tensor->dims() == 1,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "replacement_symbols must be rank 1. Got rank ",
                    replacement_symbols_tensor->dims(), ".")));

    const auto replacement_symbols =
        replacement_symbols_tensor->vec<tensorflow::tstring>();

    OP_REQUIRES(context, symbols.size() == replacement_symbols.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "symbols.shape is not equal to replacement_symbols.shape: ",
                    symbols.size(), " != ", replacement_symbols.size())));

    // (i,j,k) = the kth replaced program for symbols(j) in programs(i).
    std::vector<std::vector<std::vector<std::string>>> output_programs(
        programs.size(), std::vector<std::vector<std::string>>(
                             n_symbols, std::vector<std::string>()));

    auto DoWork = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        int sidx = i % n_symbols;
        int pidx = i / n_symbols;
        std::string symbol_to_replace = symbols(sidx);
        std::string temp_symbol_holder;
        Program cur_program = programs.at(pidx);
        for (int j = 0; j < cur_program.circuit().moments().size(); j++) {
          Moment cur_moment = cur_program.circuit().moments().at(j);
          for (int k = 0; k < cur_moment.operations().size(); k++) {
            Operation cur_op = cur_moment.operations().at(k);
            for (auto l = cur_op.args().begin(); l != cur_op.args().end();
                 l++) {
              const std::string key = (*l).first;
              const Arg &arg = (*l).second;
              if (arg.symbol() == symbol_to_replace) {
                // Copy the proto, modify the symbol and append to output.
                Program temp(cur_program);

                // temp_symbol_holder is needed to avoid call ambiguity for
                // set_symbol below.
                temp_symbol_holder = replacement_symbols(sidx);
                temp.mutable_circuit()
                    ->mutable_moments()
                    ->at(j)
                    .mutable_operations()
                    ->at(k)
                    .mutable_args()
                    ->at(key)
                    .set_symbol(temp_symbol_holder);

                std::string res;
                temp.SerializeToString(&res);
                output_programs.at(pidx).at(sidx).push_back(res);
                temp.Clear();
              }
            }
          }
        }
      }
    };

    const int block_size = GetBlockSize(context, programs.size() * n_symbols);
    context->device()
        ->tensorflow_cpu_worker_threads()
        ->workers->TransformRangeConcurrently(
            block_size, programs.size() * n_symbols, DoWork);

    size_t biggest_pad = 0;
    Program empty = Program();
    empty.mutable_language()->set_gate_set("tfq_gate_set");
    empty.mutable_circuit();  // create empty circuits entry.

    std::string empty_program;
    empty.SerializeToString(&empty_program);

    for (size_t i = 0; i < output_programs.size(); i++) {
      for (size_t j = 0; j < n_symbols; j++) {
        biggest_pad = std::max(biggest_pad, output_programs.at(i).at(j).size());
      }
    }

    tensorflow::Tensor *output = nullptr;
    tensorflow::TensorShape output_shape;
    // batch size.
    output_shape.AddDim(programs.size());
    // entry size.
    output_shape.AddDim(n_symbols);
    output_shape.AddDim(biggest_pad);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto output_tensor = output->tensor<tensorflow::tstring, 3>();

    // TODO: investigate whether or not it is worth this parallelization at the
    // end.
    //  spinning up and down parallelization for string copying might not be
    //  worth it.
    auto DoWork2 = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        int sidx = i % n_symbols;
        int pidx = i / n_symbols;
        for (int j = 0; j < output_programs.at(pidx).at(sidx).size(); j++) {
          output_tensor(pidx, sidx, j) =
              output_programs.at(pidx).at(sidx).at(j);
        }
        for (int j = output_programs.at(pidx).at(sidx).size(); j < biggest_pad;
             j++) {
          output_tensor(pidx, sidx, j) = empty_program;
        }
      }
    };
    context->device()
        ->tensorflow_cpu_worker_threads()
        ->workers->TransformRangeConcurrently(
            block_size, programs.size() * n_symbols, DoWork2);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqPsSymbolReplace").Device(tensorflow::DEVICE_CPU),
    TfqPsSymbolReplaceOp);

REGISTER_OP("TfqPsSymbolReplace")
    .Input("programs: string")
    .Input("symbols: string")
    .Input("replacement_symbols: string")
    .Output("ps_programs: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbols_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbols_shape));

      tensorflow::shape_inference::ShapeHandle replacement_symbols_shape;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(2), 1, &replacement_symbols_shape));

      c->set_output(
          0, c->MakeShape(
                 {c->Dim(programs_shape, 0),
                  tensorflow::shape_inference::InferenceContext::kUnknownDim,
                  tensorflow::shape_inference::InferenceContext::kUnknownDim}));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
