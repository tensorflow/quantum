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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "serialize",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [":serializer"],
)

py_library(
    name = "op_serializer",
    srcs = ["op_serializer.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/proto:program_py_proto"
    ],
)

py_test(
    name = "op_serializer_test",
    srcs = ["op_serializer_test.py"],
    srcs_version = "PY3",
    deps = [
        ":op_serializer",
        "//tensorflow_quantum/core/proto:program_py_proto",
    ],
)

py_library(
    name = "op_deserializer",
    srcs = ["op_deserializer.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/proto:program_py_proto"
    ],
)

py_test(
    name = "op_deserializer_test",
    srcs = ["op_deserializer_test.py"],
    srcs_version = "PY3",
    deps = [
        ":op_deserializer",
        "//tensorflow_quantum/core/proto:program_py_proto"
    ],
)

py_library(
    name = "serializable_gate_set",
    srcs = ["serializable_gate_set.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/proto:program_py_proto"
    ],
)

py_test(
    name = "serializable_gate_set_test",
    srcs = ["serializable_gate_set_test.py"],
    srcs_version = "PY3",
    deps = [
        ":op_serializer",
        ":op_deserializer",
        ":serializable_gate_set",
        "//tensorflow_quantum/core/proto:program_py_proto"
    ],
)

py_library(
    name = "serializer",
    srcs = ["serializer.py"],
    srcs_version = "PY3",
    deps = [
        # cirq proto
        ":op_serializer",
        ":op_deserializer",
        ":serializable_gate_set",
        "//tensorflow_quantum/core/proto:pauli_sum_py_proto",
        "//tensorflow_quantum/core/proto:program_py_proto",
        "//tensorflow_quantum/core/proto:projector_sum_py_proto",
    ],
)

py_test(
    name = "serializer_test",
    srcs = ["serializer_test.py"],
    python_version = "PY3",
    deps = [
        ":serializer",
        # cirq proto
        "//tensorflow_quantum/core/proto:pauli_sum_py_proto",
        "//tensorflow_quantum/core/proto:projector_sum_py_proto",
    ],
)
