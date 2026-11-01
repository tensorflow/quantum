/* Copyright 2026 The TensorFlow Quantum Authors. All Rights Reserved.

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

#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"

#include <limits>

#include "gtest/gtest.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"

namespace tfq {
namespace {

class FakeCpuDevice : public tensorflow::DeviceBase {
 public:
  FakeCpuDevice() : tensorflow::DeviceBase(tensorflow::Env::Default()) {}
};

class NullWorkerThreadsDevice : public tensorflow::DeviceBase {
 public:
  NullWorkerThreadsDevice()
      : tensorflow::DeviceBase(tensorflow::Env::Default()) {}

  const CpuWorkerThreads* tensorflow_cpu_worker_threads() const override {
    return nullptr;
  }
};

TEST(TfqSimulateUtilsTest, NullContextReturnsOne) {
  EXPECT_EQ(GetBlockSize(nullptr, 100), 1);
  EXPECT_EQ(GetBlockSize(nullptr, 0), 1);
  EXPECT_EQ(GetBlockSize(nullptr, -5), 1);
  EXPECT_EQ(GetBlockSize(nullptr, std::numeric_limits<int>::min()), 1);
}

TEST(TfqSimulateUtilsTest, NullDeviceReturnsOne) {
  // OpKernelContext requires a non-null device during construction, and
  // context.device() reads params.device dynamically afterward.
  FakeCpuDevice device;
  tensorflow::OpKernelContext::Params params;
  params.device = &device;
  tensorflow::OpKernelContext context(&params, 0);
  params.device = nullptr;

  EXPECT_EQ(GetBlockSize(&context, 100), 1);
  EXPECT_EQ(GetBlockSize(&context, 0), 1);
  EXPECT_EQ(GetBlockSize(&context, -5), 1);
}

TEST(TfqSimulateUtilsTest, NonPositiveOutputSizeShortCircuitsBeforeThreads) {
  // DeviceBase::tensorflow_cpu_worker_threads() CHECK-aborts if unconfigured,
  // proving non-positive output_size returns before querying worker threads.
  FakeCpuDevice unconfigured_device;
  tensorflow::OpKernelContext::Params params;
  params.device = &unconfigured_device;
  tensorflow::OpKernelContext context(&params, 0);

  EXPECT_EQ(GetBlockSize(&context, 0), 1);
  EXPECT_EQ(GetBlockSize(&context, -5), 1);
  EXPECT_EQ(GetBlockSize(&context, std::numeric_limits<int>::min()), 1);
}

TEST(TfqSimulateUtilsTest, NullWorkerThreadsReturnsOne) {
  NullWorkerThreadsDevice device;
  tensorflow::OpKernelContext::Params params;
  params.device = &device;
  tensorflow::OpKernelContext context(&params, 0);

  EXPECT_EQ(GetBlockSize(&context, 100), 1);
}

TEST(TfqSimulateUtilsTest, ZeroOrNegativeThreadsReturnsOne) {
  tensorflow::DeviceBase::CpuWorkerThreads worker_threads;
  FakeCpuDevice device;
  device.set_tensorflow_cpu_worker_threads(&worker_threads);

  tensorflow::OpKernelContext::Params params;
  params.device = &device;
  tensorflow::OpKernelContext context(&params, 0);

  worker_threads.num_threads = 0;
  EXPECT_EQ(GetBlockSize(&context, 100), 1);

  worker_threads.num_threads = -2;
  EXPECT_EQ(GetBlockSize(&context, 100), 1);

  worker_threads.num_threads = std::numeric_limits<int>::min();
  EXPECT_EQ(GetBlockSize(&context, 100), 1);
}

TEST(TfqSimulateUtilsTest, ValidInputsPreserveFloorDivisionClampedToOne) {
  tensorflow::DeviceBase::CpuWorkerThreads worker_threads;
  FakeCpuDevice device;
  worker_threads.num_threads = 4;
  device.set_tensorflow_cpu_worker_threads(&worker_threads);

  tensorflow::OpKernelContext::Params params;
  params.device = &device;
  tensorflow::OpKernelContext context(&params, 0);

  EXPECT_EQ(GetBlockSize(&context, 100), 25);
  EXPECT_EQ(GetBlockSize(&context, 8), 2);
  EXPECT_EQ(GetBlockSize(&context, 7), 1);
  EXPECT_EQ(GetBlockSize(&context, 4), 1);
  EXPECT_EQ(GetBlockSize(&context, 3), 1);
  EXPECT_EQ(GetBlockSize(&context, 0), 1);
  EXPECT_EQ(GetBlockSize(&context, -5), 1);
  EXPECT_EQ(GetBlockSize(&context, 2147483647), 536870911);
}

TEST(TfqSimulateUtilsTest, SingleWorkerThreadReturnsFullOutputSize) {
  tensorflow::DeviceBase::CpuWorkerThreads worker_threads;
  FakeCpuDevice device;
  worker_threads.num_threads = 1;
  device.set_tensorflow_cpu_worker_threads(&worker_threads);

  tensorflow::OpKernelContext::Params params;
  params.device = &device;
  tensorflow::OpKernelContext context(&params, 0);

  EXPECT_EQ(GetBlockSize(&context, 100), 100);
}

}  // namespace
}  // namespace tfq
