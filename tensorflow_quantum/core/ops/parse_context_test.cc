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

#include "tensorflow_quantum/core/ops/parse_context.h"

#include "absl/synchronization/mutex.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tfq {
namespace {

TEST(ParseContextTest, NestedFnStatusSyncEvaluatesOnce) {
  tensorflow::Status global_status;
  absl::Mutex global_lock;

  int ok_evals = 0;
  bool ok_completed = false;
  auto ok_fn = [&]() {
    NESTED_FN_STATUS_SYNC(global_status, ([&]() {
                            ++ok_evals;
                            return tensorflow::Status();
                          })(),
                          global_lock);
    ok_completed = true;
  };
  ok_fn();
  EXPECT_EQ(ok_evals, 1);
  EXPECT_TRUE(global_status.ok());
  EXPECT_TRUE(ok_completed);

  int err_evals = 0;
  bool err_completed = false;
  auto err_fn = [&]() {
    NESTED_FN_STATUS_SYNC(
        global_status, ([&]() {
          ++err_evals;
          return tensorflow::errors::InvalidArgument("test error");
        })(),
        global_lock);
    err_completed = true;
  };
  err_fn();
  EXPECT_EQ(err_evals, 1);
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(global_status));
  EXPECT_FALSE(err_completed);
}

TEST(ParseContextTest, NestedFnStatusSyncAvoidsCallerNameCollisions) {
  tensorflow::Status _s;
  absl::Mutex _l;
  bool completed_after_ok = false;
  auto ok_fn = [&]() {
    NESTED_FN_STATUS_SYNC(_s, tensorflow::Status(), _l);
    completed_after_ok = true;
  };
  ok_fn();
  EXPECT_TRUE(_s.ok());
  EXPECT_TRUE(completed_after_ok);

  bool completed_after_err = false;
  auto err_fn = [&]() {
    NESTED_FN_STATUS_SYNC(
        _s, tensorflow::errors::InvalidArgument("collision error"), _l);
    completed_after_err = true;
  };
  err_fn();
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(_s));
  EXPECT_FALSE(completed_after_err);

  tensorflow::Status status;
  absl::Mutex lock;
  auto read_from_s_fn = [&]() { NESTED_FN_STATUS_SYNC(status, _s, lock); };
  read_from_s_fn();
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(status));
}

}  // namespace
}  // namespace tfq
