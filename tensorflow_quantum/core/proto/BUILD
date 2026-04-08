# Copyright 2020 The TensorFlow Quantum Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

proto_library(
    name = "program_proto",
    srcs = ["program.proto"]
)

py_proto_library(
    name = "program_py_proto",
    srcs = ["program.proto"],
)

cc_proto_library(
    name = "program_cc_proto",
    deps = [":program_proto"],
)

py_proto_library(
    name = "pauli_sum_py_proto",
    srcs = ["pauli_sum.proto"],
)

proto_library(
    name = "pauli_sum_proto",
    srcs = ["pauli_sum.proto"],
)

cc_proto_library(
    name = "pauli_sum_cc_proto",
    deps = [
        ":pauli_sum_proto",
    ],
)

py_proto_library(
    name = "projector_sum_py_proto",
    srcs = ["projector_sum.proto"],
)

proto_library(
    name = "projector_sum_proto",
    srcs = ["projector_sum.proto"],
)

cc_proto_library(
    name = "projector_sum_cc_proto",
    deps = [
        ":projector_sum_proto",
    ],
)
