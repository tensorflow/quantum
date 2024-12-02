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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"
#include "tensorflow_quantum/core/proto/program.pb.h"

namespace tfq {

using ::tensorflow::Tensor;
using ::tfq::proto::Arg;
using ::tfq::proto::Moment;
using ::tfq::proto::Operation;
using ::tfq::proto::Program;

class TfqPsWeightsFromSymbolOp : public tensorflow::OpKernel {
 public:
  explicit TfqPsWeightsFromSymbolOp(tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override {
    std::vector<Program> programs;

    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 2,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 2 inputs, got ", num_inputs, " inputs.")));

    OP_REQUIRES_OK(context, ParsePrograms(context, "programs", &programs));

    // Parse the input string here.
    const Tensor *symbols_tensor;
    OP_REQUIRES_OK(context, context->input("symbols", &symbols_tensor));
    OP_REQUIRES(
        context, symbols_tensor->dims() == 1,
        tensorflow::errors::InvalidArgument(absl::StrCat(
            "symbols must be rank 1. Got rank ", symbols_tensor->dims(), ".")));

    const auto symbols = symbols_tensor->vec<tensorflow::tstring>();
    const int n_symbols = symbols.size();

    // (i,j,k) = the kth scalar value found for symbols(j) in programs(i).
    std::vector<std::vector<std::vector<float>>> output_results(
        programs.size(),
        std::vector<std::vector<float>>(n_symbols, std::vector<float>()));

    // map from symbols -> index in second dimension of output_results.
    absl::flat_hash_map<std::string, int> symbols_map;
    for (int i = 0; i < n_symbols; i++) {
      symbols_map[symbols(i)] = i;
    }
    std::vector<std::string> ignore_list = {"I",  "ISP", "PXP", "FSIM", "PISP",
                                            "AD", "ADP", "DP",  "GAD",  "BF",
                                            "PF", "PD",  "RST"};
    absl::flat_hash_set<std::string> ignored_symbol_set(ignore_list.begin(),
                                                        ignore_list.end());

    std::vector<int> n_single_symbol(programs.size(), 0);

    auto DoWork = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        Program cur_program = programs.at(i);
        for (int j = 0; j < cur_program.circuit().moments().size(); j++) {
          Moment cur_moment = cur_program.circuit().moments().at(j);
          for (int k = 0; k < cur_moment.operations().size(); k++) {
            Operation cur_op = cur_moment.operations().at(k);
            if (ignored_symbol_set.contains(cur_op.gate().id())) continue;

            const auto &cur_op_map = *cur_op.mutable_args();
            const auto exponent = cur_op_map.at("exponent");
            if (exponent.arg_case() == Arg::ArgCase::kSymbol) {
              // this gate has parameterized exponent.
              const absl::string_view symbol_name = exponent.symbol();
              if (!symbols_map.contains(symbol_name)) {
                // Should never happen. raise error.
                OP_REQUIRES(context, false,
                            tensorflow::errors::InvalidArgument(
                                "A circuit contains a sympy.Symbol not found "
                                "in symbols!"));
              }
              output_results.at(i)
                  .at(symbols_map.at(symbol_name))
                  .push_back(cur_op.args()
                                 .at("exponent_scalar")
                                 .arg_value()
                                 .float_value());
            }
          }
        }
        // loop over all index entries of symbols_map and find largest
        // value from output_results.
        for (int j = 0; j < n_symbols; j++) {
          n_single_symbol.at(i) =
              std::max(n_single_symbol.at(i),
                       static_cast<int>(output_results.at(i).at(j).size()));
        }
      }
    };

    const int block_size = GetBlockSize(context, programs.size());
    context->device()
        ->tensorflow_cpu_worker_threads()
        ->workers->TransformRangeConcurrently(block_size, programs.size(),
                                              DoWork);

    int largest_single_symbol = 0;
    for (size_t i = 0; i < n_single_symbol.size(); i++) {
      largest_single_symbol =
          std::max(n_single_symbol.at(i), largest_single_symbol);
    }

    tensorflow::Tensor *output = nullptr;
    tensorflow::TensorShape output_shape;
    // batch size.
    output_shape.AddDim(programs.size());
    // entry size.
    output_shape.AddDim(n_symbols);
    output_shape.AddDim(largest_single_symbol);

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto output_tensor = output->tensor<float, 3>();

    auto DoWork2 = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        for (int j = 0; j < n_symbols; j++) {
          for (std::vector<float>::size_type k = 0;
               k < output_results.at(i).at(j).size(); k++) {
            output_tensor(i, j, k) = output_results.at(i).at(j).at(k);
          }
          for (int k = output_results.at(i).at(j).size();
               k < largest_single_symbol; k++) {
            output_tensor(i, j, k) = 0.0f;
          }
        }
      }
    };
    context->device()
        ->tensorflow_cpu_worker_threads()
        ->workers->TransformRangeConcurrently(block_size, programs.size(),
                                              DoWork2);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqPsWeightsFromSymbols").Device(tensorflow::DEVICE_CPU),
    TfqPsWeightsFromSymbolOp);

REGISTER_OP("TfqPsWeightsFromSymbols")
    .Input("programs: string")
    .Input("symbols: string")
    .Output("weights: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbols_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbols_shape));

      c->set_output(
          0, c->MakeShape(
                 {c->Dim(programs_shape, 0),
                  tensorflow::shape_inference::InferenceContext::kUnknownDim,
                  tensorflow::shape_inference::InferenceContext::kUnknownDim}));

      return ::tensorflow::Status();
    });

}  // namespace tfq
