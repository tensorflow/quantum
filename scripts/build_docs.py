# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
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
# ==============================================================================
"""Tool to generate external api_docs for tfq."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_quantum as tfq

flags.DEFINE_string("output_dir", "/tmp/tfq_api", "Where to output the docs")

flags.DEFINE_string("code_url_prefix",
                    ("https://github.com/tensorflow/quantum/tree/master/"
                     "tensorflow_quantum"), "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "quantum/api_docs/python",
                    "Path prefix in the _toc.yaml")

FLAGS = flags.FLAGS




def main(unused_argv):

    doc_generator = generate_lib.DocGenerator(
        root_title="TensorFlow Quantum",
        py_modules=[("tfq", tfq)],
        base_dir=os.path.dirname(tfq.__file__),
        code_url_prefix=FLAGS.code_url_prefix,
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[public_api.local_definitions_filter],
        private_map={
            "tfq": ["python", "core"],
            "tfq.layers": [
                "circuit_construction",
                "circuit_executors",
                "high_level",
            ],
            "tfq.differentiators": [
                "linear_combination", "differentiator", "parameter_shift",
                "stochastic_differentiator", "parameter_shift_util",
                "stochastic_differentiator_util"
            ],
            "tfq.datasets": ["cluster_state"],
            "tfq.util": ["from_tensor", "convert_to_tensor",
                        "exp_identity", "check_commutability", "kwargs_cartesian_product", "random_circuit_resolver_batch", "random_pauli_sums", "random_symbol_circuit", "random_symbol_circuit_resolver_batch"]
        })

    doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
