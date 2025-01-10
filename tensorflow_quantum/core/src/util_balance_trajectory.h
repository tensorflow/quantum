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

#ifndef UTIL_BALANCE_TRAJECTORY_H_
#define UTIL_BALANCE_TRAJECTORY_H_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tfq {

void BalanceTrajectory(const std::vector<std::vector<int>>& num_samples,
                       const int& num_threads,
                       std::vector<std::vector<int>>* thread_offsets);

void BalanceTrajectory(const int& num_samples, const int& num_threads,
                       std::vector<std::vector<int>>* thread_offsets);

}  // namespace tfq

#endif  // UTIL_BALANCE_TRAJECTORY_H_
