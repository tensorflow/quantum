licenses(["notice"])

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        # Include the __init__.py files
        # Module level
        "__init__.py",
        "//tensorflow_quantum:__init__.py",

        # Core module.
        "//tensorflow_quantum/core:__init__.py",
        "//tensorflow_quantum/core/ops:__init__.py",
        "//tensorflow_quantum/core/ops/math_ops:__init__.py",
        "//tensorflow_quantum/core/ops/noise:__init__.py",
        "//tensorflow_quantum/core/proto:__init__.py",
        "//tensorflow_quantum/core/serialize:__init__.py",

        # Python module.
        "//tensorflow_quantum/python:__init__.py",
        "//tensorflow_quantum/python/differentiators:__init__.py",
        "//tensorflow_quantum/python/layers:__init__.py",
        "//tensorflow_quantum/python/layers/circuit_construction:__init__.py",
        "//tensorflow_quantum/python/layers/circuit_executors:__init__.py",
        "//tensorflow_quantum/python/layers/high_level:__init__.py",
        "//tensorflow_quantum/python/optimizers:__init__.py",

        # Datasets module.
        "//tensorflow_quantum/datasets:__init__.py",

        # Dependencies to run PIP package creation.
        "MANIFEST.in",
        "setup.py",

        # Dependencies that the PIP package will import.
        "//tensorflow_quantum/core/ops:tfq_adj_grad_op_py",
        "//tensorflow_quantum/core/ops:tfq_ps_util_ops_py",
        "//tensorflow_quantum/core/ops:tfq_unitary_op_py",
        "//tensorflow_quantum/core/ops:tfq_utility_ops_py",
        "//tensorflow_quantum/core/ops:tfq_simulate_ops_py",
        "//tensorflow_quantum/core/ops/math_ops:inner_product_op_py",
        "//tensorflow_quantum/core/ops/math_ops:fidelity_op_py",
        "//tensorflow_quantum/core/ops/noise:noisy_samples_op_py",
        "//tensorflow_quantum/core/ops/noise:noisy_expectation_op_py",
        "//tensorflow_quantum/core/ops/noise:noisy_sampled_expectation_op_py",
        "//tensorflow_quantum/core/serialize:serializer",
        "//tensorflow_quantum/datasets:cluster_state",
        "//tensorflow_quantum/datasets:spin_system",
        "//tensorflow_quantum/python/differentiators:adjoint",
        "//tensorflow_quantum/python/differentiators:differentiator",
        "//tensorflow_quantum/python/differentiators:parameter_shift",
        "//tensorflow_quantum/python/differentiators:linear_combination",
        "//tensorflow_quantum/python/layers/circuit_construction:elementary",
        "//tensorflow_quantum/python/layers/circuit_executors:expectation",
        "//tensorflow_quantum/python/layers/circuit_executors:sample",
        "//tensorflow_quantum/python/layers/circuit_executors:state",
        "//tensorflow_quantum/python/layers/circuit_executors:sampled_expectation",
        "//tensorflow_quantum/python/layers/circuit_executors:unitary",
        "//tensorflow_quantum/python/layers/high_level:controlled_pqc",
        "//tensorflow_quantum/python/layers/high_level:noisy_pqc",
        "//tensorflow_quantum/python/layers/high_level:pqc",
        "//tensorflow_quantum/python:quantum_context",
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/optimizers:rotosolve_minimizer",
    ],
)
