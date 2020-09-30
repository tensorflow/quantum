package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])


py_binary(
	name = "speed_test",
	srcs = ["speed_test.py"],
	deps = ["//tensorflow_quantum/python:util",
			"//tensorflow_quantum/core/ops:tfq_utility_ops_py",
			"//tensorflow_quantum/core/ops/math_ops:inner_product_op_py",
			"//tensorflow_quantum/python/layers/circuit_executors:expectation",
			"//tensorflow_quantum/python/layers/circuit_construction:elementary"]

)