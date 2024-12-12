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

#include "gtest/gtest.h"


namespace tfq {
namespace {

static void AssertWellBalanced(const std::vector<std::vector<int>>& n_reps,
                               const int& num_threads,
                               const std::vector<std::vector<int>>& offsets) {
  auto max_work = std::vector<int>(n_reps.size(), -1);
  for (size_t i = 0; i < n_reps.size(); i++) {
    for (size_t j = 0; j < n_reps[0].size(); j++) {
      max_work[i] = std::max(max_work[i], n_reps[i][j]);
    }
  }

  for (size_t i = 0; i < n_reps.size(); i++) {
    int sum = 0;
    int prev_local_work = 0;
    for (int k = 0; k < num_threads; k++) {
      int local_work = (max_work[i] + num_threads - 1) / num_threads;
      local_work += offsets[k][i];
      sum += local_work;
      if (k > 0) {
        EXPECT_LT(abs(local_work - prev_local_work), 2);
      }
      prev_local_work = local_work;
    }
    EXPECT_EQ(sum, max_work[i]);
  }
}

TEST(UtilQsimTest, BalanceTrajectorySimple) {
  std::vector<std::vector<int>> n_reps = {{1, 3, 5, 10, 15},
                                          {1, 10, 20, 30, 40},
                                          {50, 70, 100, 100, 100},
                                          {100, 200, 200, 200, 200}};
  const int num_threads = 3;
  // [num_threads, n_reps.size()]
  std::vector<std::vector<int>> offsets = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

  BalanceTrajectory(n_reps, num_threads, &offsets);
  AssertWellBalanced(n_reps, num_threads, offsets);
}

TEST(UtilQsimTest, BalanceTrajectoryPreventIdle) {
  std::vector<std::vector<int>> n_reps = {{1, 1, 1, 1, 11},
                                          {1, 1, 1, 11, 1},
                                          {1, 1, 11, 1, 1},
                                          {1, 11, 1, 1, 1},
                                          {11, 1, 1, 1, 1}};
  const int num_threads = 10;
  // [num_threads, n_reps.size()]
  std::vector<std::vector<int>> offsets = {
      {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};

  BalanceTrajectory(n_reps, num_threads, &offsets);
  AssertWellBalanced(n_reps, num_threads, offsets);
}

TEST(UtilQsimTest, BalanceTrajectoryLowRep) {
  std::vector<std::vector<int>> n_reps = {
      {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  const int num_threads = 5;
  // [num_threads, n_reps.size()]
  std::vector<std::vector<int>> offsets = {{0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0}};

  BalanceTrajectory(n_reps, num_threads, &offsets);
  AssertWellBalanced(n_reps, num_threads, offsets);
}

TEST(UtilQsimTest, BalanceTrajectoryFewHigh) {
  std::vector<std::vector<int>> n_reps = {
      {1, 100, 1, 1, 1}, {1, 1, 1, 1, 1000}, {1, 1, 1, 1, 1},   {1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1},   {1, 10, 1, 1, 1},   {1, 1, 1, 1, 1000}};
  const int num_threads = 5;
  // [num_threads, n_reps.size()]
  std::vector<std::vector<int>> offsets = {{0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0}};

  BalanceTrajectory(n_reps, num_threads, &offsets);
  AssertWellBalanced(n_reps, num_threads, offsets);
}

TEST(UtilQsimTest, BalanceTrajectory1D) {
  const int n_reps = 100;
  const int num_threads = 5;
  // [num_threads, batch_size]
  std::vector<std::vector<int>> offsets = {{0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0, 0}};

  std::vector<std::vector<int>> tmp(offsets[0].size(),
                                    std::vector<int>(2, n_reps));
  BalanceTrajectory(n_reps, num_threads, &offsets);
  AssertWellBalanced(tmp, num_threads, offsets);
}

TEST(UtilQsimTest, BalanceTrajectory1D_2) {
  const int n_reps = 11;
  const int num_threads = 10;
  // [num_threads, batch_size]
  std::vector<std::vector<int>> offsets = {
      {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};

  std::vector<std::vector<int>> tmp(offsets[0].size(),
                                    std::vector<int>(2, n_reps));
  BalanceTrajectory(n_reps, num_threads, &offsets);
  AssertWellBalanced(tmp, num_threads, offsets);
}

}  // namespace
}  // namespace tfq
