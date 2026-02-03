# CPU-only stub for TensorFlow/XLA builds that attempt to load CUDA build defs
# during analysis. TFQ is building CPU-only, so these must be safe no-ops.

def if_cuda(then_labels, else_labels = []):
    return else_labels

def cuda_default_copts():
    return []

def cuda_is_configured():
    return False
