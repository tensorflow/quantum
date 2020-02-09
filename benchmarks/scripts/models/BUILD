package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_binary(
    name = "random_clifford_circuit",
    srcs = ["random_clifford_circuit.py"],
    python_version = "PY3",
)

py_test(
    name = "random_clifford_circuit_test",
    srcs = ["random_clifford_circuit_test.py"],
    python_version = "PY3",
    deps = [
        ":random_clifford_circuit",
    ],
)
