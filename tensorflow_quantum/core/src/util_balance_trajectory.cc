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

#include "tensorflow_quantum/core/src/util_balance_trajectory.h"

namespace tfq {

// Balance the number of trajectory computations done between
// threads. num_samples is a 2d vector containing the number of reps
// requested for each pauli_sum[i,j]. After running thread_offsets
// contains 0/-1 values that will offset the work for each thread.
// to make it as close to uniform as possible. **Assumes circuits
// have roughly equal simulation cost**
void BalanceTrajectory(const std::vector<std::vector<int>>& num_samples,
                       const int& num_threads,
                       std::vector<std::vector<int>>* thread_offsets) {
  std::vector<int> rep_limits(num_samples.size(), -1);
  std::vector<int> height(num_threads, 0);

  for (size_t i = 0; i < num_samples.size(); i++) {
    for (size_t j = 0; j < num_samples[i].size(); j++) {
      rep_limits[i] = std::max(rep_limits[i], num_samples[i][j]);
    }
  }
  int prev_max_height = -1;
  for (size_t j = 0; j < num_samples.size(); j++) {
    int run_ceiling = ((rep_limits[j] + num_threads - 1) / num_threads);
    int num_lo = num_threads * run_ceiling - rep_limits[j];
    int num_hi = num_threads - num_lo;
    int cur_max = prev_max_height;
    for (int i = 0; i < num_threads; i++) {
      if (height[i] == cur_max && num_lo) {
        // previously had extra work on this thread and
        // have remaining low budget to give.
        height[i]++;
        (*thread_offsets)[i][j] = -1;
        num_lo--;
      } else if (height[i] == cur_max - 1 && num_hi) {
        // previously had less work on this thread and
        // remaining high budget to give.
        height[i] += 2;
        (*thread_offsets)[i][j] = 0;
        num_hi--;
      } else if (num_hi) {
        height[i] += 2;
        (*thread_offsets)[i][j] = 0;
        num_hi--;
      } else {
        height[i]++;
        (*thread_offsets)[i][j] = -1;
        num_lo--;
      }
      prev_max_height = std::max(height[i], prev_max_height);
    }
  }
}

// Simpler case of TrajectoryBalance where num_samples is fixed
// across all circuits.
void BalanceTrajectory(const int& num_samples, const int& num_threads,
                       std::vector<std::vector<int>>* thread_offsets) {
  std::vector<int> height(num_threads, 0);

  int prev_max_height = -1;
  for (size_t j = 0; j < (*thread_offsets)[0].size(); j++) {
    int run_ceiling = ((num_samples + num_threads - 1) / num_threads);
    int num_lo = num_threads * run_ceiling - num_samples;
    int num_hi = num_threads - num_lo;
    int cur_max = prev_max_height;
    for (int i = 0; i < num_threads; i++) {
      if (height[i] == cur_max && num_lo) {
        // previously had extra work on this thread and
        // have remaining low budget to give.
        height[i]++;
        (*thread_offsets)[i][j] = -1;
        num_lo--;
      } else if (height[i] == cur_max - 1 && num_hi) {
        // previously had less work on this thread and
        // remaining high budget to give.
        height[i] += 2;
        (*thread_offsets)[i][j] = 0;
        num_hi--;
      } else if (num_hi) {
        height[i] += 2;
        (*thread_offsets)[i][j] = 0;
        num_hi--;
      } else {
        height[i]++;
        (*thread_offsets)[i][j] = -1;
        num_lo--;
      }
      prev_max_height = std::max(height[i], prev_max_height);
    }
  }
}

}  // namespace tfq
