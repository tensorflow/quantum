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

#include <cmath>
#include <string>

#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"

namespace tfq {

using ::cirq::google::api::v2::Arg;
using ::cirq::google::api::v2::Circuit;
using ::cirq::google::api::v2::Moment;
using ::cirq::google::api::v2::Operation;
using ::cirq::google::api::v2::Program;
using ::cirq::google::api::v2::Qubit;
using ::tensorflow::Status;
using ::tensorflow::Tensor;

class TfqPsDecomposeOp : public tensorflow::OpKernel {
 public:
  explicit TfqPsDecomposeOp(tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override {
    std::vector<Program> programs;

    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 1,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 1 inputs, got ", num_inputs, " inputs.")));

    OP_REQUIRES_OK(context, ParsePrograms(context, "programs", &programs));

    tensorflow::Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, context->input(0).shape(), &output));
    auto output_tensor = output->flat<std::string>();

    const int max_buffer_moments = 2;

    auto DoWork = [&](int start, int end) {
      for (int i = start; i < end; i++) {
        Program cur_program = programs.at(i);
        Program new_program;
        new_program.mutable_language()->set_gate_set("tfq_gate_set");
        new_program.mutable_circuit()->set_scheduling_strategy(
            Circuit::MOMENT_BY_MOMENT);
        for (int j = 0; j < cur_program.circuit().moments().size(); j++) {
          Moment cur_moment(cur_program.circuit().moments().at(j));
          std::vector<Moment> temp_moment_list(max_buffer_moments, Moment());
          int num_extra_moments = 0;
          for (int k = 0; k < cur_moment.operations().size(); k++) {
            Operation cur_op = cur_moment.operations().at(k);
            auto &cur_op_map = *cur_op.mutable_args();
            if (cur_op.gate().id() == "ISP") {
              auto exponent = cur_op_map.at("exponent");
              if (exponent.arg_case() == Arg::ArgCase::kSymbol) {
                // Decompose cirq.ISwapPowGate only if it is parameterized.
                if (num_extra_moments == 0) num_extra_moments = 1;
                Operation new_op;
                new_op = getOpForISP(cur_op, "XXP", exponent.symbol());
                cur_moment.mutable_operations()->at(k) = new_op;
                new_op = getOpForISP(cur_op, "YYP", exponent.symbol());
                *temp_moment_list[0].add_operations() = new_op;
              }
            } else if (cur_op.gate().id() == "PXP") {
              auto exponent = cur_op_map.at("exponent");
              auto phase_exponent = cur_op_map.at("phase_exponent");
              if (exponent.arg_case() == Arg::ArgCase::kSymbol ||
                  phase_exponent.arg_case() == Arg::ArgCase::kSymbol) {
                // Decompose cirq.PhasedXPowGate only if it is parameterized.
                num_extra_moments = 2;
                Operation new_op;
                new_op = getOpForPXP(cur_op, "ZP", "phase_exponent", true);
                cur_moment.mutable_operations()->at(k) = new_op;
                new_op = getOpForPXP(cur_op, "XP", "exponent", false);
                *temp_moment_list[0].add_operations() = new_op;
                new_op = getOpForPXP(cur_op, "ZP", "phase_exponent", false);
                *temp_moment_list[1].add_operations() = new_op;
              }
            } else if (cur_op.gate().id() == "FSIM") {
              auto theta = cur_op_map.at("theta");
              auto phi = cur_op_map.at("phi");
              if (theta.arg_case() == Arg::ArgCase::kSymbol ||
                  phi.arg_case() == Arg::ArgCase::kSymbol) {
                // Decompose cirq.FSimGate only if it is parameterized.
                num_extra_moments = 2;
                Operation new_op;
                new_op = getOpForFSIM(cur_op, "XXP", "theta", true);
                cur_moment.mutable_operations()->at(k) = new_op;
                new_op = getOpForFSIM(cur_op, "YYP", "theta", true);
                *temp_moment_list[0].add_operations() = new_op;
                new_op = getOpForFSIM(cur_op, "CZP", "phi", false);
                *temp_moment_list[1].add_operations() = new_op;
              }
            }
          }
          *new_program.mutable_circuit()->add_moments() = cur_moment;
          if (num_extra_moments > 0) {
            for (int l = 0; l < num_extra_moments; l++) {
              *new_program.mutable_circuit()->add_moments() =
                  temp_moment_list[l];
            }
          }
        }
        new_program.SerializeToString(&output_tensor(i));
      }
    };

    const int block_size = GetBlockSize(context, programs.size());
    context->device()
        ->tensorflow_cpu_worker_threads()
        ->workers->TransformRangeConcurrently(block_size, programs.size(),
                                              DoWork);
  }

 private:
  // Helper functions for decompositions of ISwapPowGate, PhasedX
  Operation getOpForISP(Operation &cur_op, std::string id, std::string symbol) {
    // Step 1. parse the current op.
    auto &cur_op_map = *cur_op.mutable_args();
    float cur_exponent_scalar =
        cur_op_map["exponent_scalar"].arg_value().float_value();
    auto &cur_op_qubits = cur_op.qubits();
    // Step 2. create a new op.
    Operation new_op;
    new_op.mutable_gate()->set_id(id);
    // Step 3. add global_shift, exponent_scalar, exponent.
    auto &new_op_map = *new_op.mutable_args();
    new_op_map["global_shift"].mutable_arg_value()->set_float_value(-0.5);
    new_op_map["exponent_scalar"].mutable_arg_value()->set_float_value(
        cur_exponent_scalar * -0.5);
    new_op_map["exponent"].set_symbol(symbol);
    // Step 4. add qubits.
    *new_op.mutable_qubits() = {cur_op_qubits.begin(), cur_op_qubits.end()};
    return new_op;
  }

  Operation getOpForPXP(Operation &cur_op, std::string id, std::string key,
                        bool sign_flip = false) {
    // Step 1. parse the current op.
    auto &cur_op_map = *cur_op.mutable_args();
    auto &cur_op_qubits = cur_op.qubits();
    auto target_exponent = cur_op_map[key];
    float target_exponent_scalar =
        cur_op_map[absl::StrCat(key, "_scalar")].arg_value().float_value();
    float sign = (sign_flip) ? -1.0 : 1.0;
    // Step 2. create a new op.
    Operation new_op;
    new_op.mutable_gate()->set_id(id);
    // Step 3. add global_shift, exponent_scalar, exponent.
    auto &new_op_map = *new_op.mutable_args();
    new_op_map["global_shift"].mutable_arg_value()->set_float_value(0.0);
    switch (target_exponent.arg_case()) {
      case Arg::ArgCase::kSymbol:
        new_op_map["exponent_scalar"].mutable_arg_value()->set_float_value(
            sign * target_exponent_scalar);
        new_op_map["exponent"].set_symbol(target_exponent.symbol());
        break;
      case Arg::ArgCase::kArgValue:
        new_op_map["exponent_scalar"].mutable_arg_value()->set_float_value(1.0);
        new_op_map["exponent"].mutable_arg_value()->set_float_value(
            sign * target_exponent.arg_value().float_value());
        break;
      case Arg::ArgCase::kFunc:
        // TODO(jaeyoo) : support this if prepared.
        break;
      default:
        break;
    }
    // Step 4. add qubits.
    *new_op.mutable_qubits() = {cur_op_qubits.begin(), cur_op_qubits.end()};
    return new_op;
  }

  Operation getOpForFSIM(Operation &cur_op, std::string id, std::string key,
                         bool use_global_shift = false) {
    // Step 1. parse the current op.
    auto &cur_op_map = *cur_op.mutable_args();
    auto &cur_op_qubits = cur_op.qubits();
    auto target_exponent = cur_op_map[key];
    float target_exponent_scalar =
        cur_op_map[absl::StrCat(key, "_scalar")].arg_value().float_value();
    float global_shift = (use_global_shift) ? -0.5 : 0.0;
    float sign = (key == "theta") ? 1.0 : -1.0;
    // Step 2. create a new op.
    Operation new_op;
    new_op.mutable_gate()->set_id(id);
    // Step 3. add global_shift, exponent_scalar, exponent.
    auto &new_op_map = *new_op.mutable_args();
    new_op_map["global_shift"].mutable_arg_value()->set_float_value(
        global_shift);
    switch (target_exponent.arg_case()) {
      case Arg::ArgCase::kSymbol:
        new_op_map["exponent_scalar"].mutable_arg_value()->set_float_value(
            sign * target_exponent_scalar / M_PI);
        new_op_map["exponent"].set_symbol(target_exponent.symbol());
        break;
      case Arg::ArgCase::kArgValue:
        new_op_map["exponent_scalar"].mutable_arg_value()->set_float_value(1.0);
        new_op_map["exponent"].mutable_arg_value()->set_float_value(
            sign * target_exponent.arg_value().float_value() / M_PI);
        break;
      case Arg::ArgCase::kFunc:
        // TODO(jaeyoo) : support this if prepared.
        break;
      default:
        break;
    }
    // Step 4. add qubits.
    *new_op.mutable_qubits() = {cur_op_qubits.begin(), cur_op_qubits.end()};
    return new_op;
  }
};

REGISTER_KERNEL_BUILDER(Name("TfqPsDecompose").Device(tensorflow::DEVICE_CPU),
                        TfqPsDecomposeOp);

REGISTER_OP("TfqPsDecompose")
    .Input("programs: string")
    .Output("ps_programs: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      c->set_output(0, c->input(0));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
