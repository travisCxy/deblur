#licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

#load("//tensorflow_addons:tensorflow_addons.bzl", "custom_op_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_kernel_library") 

cc_library(
    name = "cub",
    hdrs = glob(["cub/**"]),
#    include_prefix = "third_party",
#    deps = [
#        "@local_config_cuda//cuda:cuda_headers",
#    ],
)

tf_kernel_library(
    name = "attention_decoder_op.so",
#    hdrs = [
#         "cc/kernels/beam_search_ops.h",
#    ],
#    hdrs = glob(["cub/**"]),
    srcs = [
        "attention_decoder.h",
        "attention_decoder.cc",
        "attention_decoder_reg.cc",
#        "blas_gemm.cc",
    ],
    gpu_srcs = [
        "attention_decoder.h",
        "topk_gpu.h",
        "reduction_gpu_kernels.cu.h",
#        "reduction_ops_gpu_float.cu.cc",
        "attention_decoder_gpu.cu.cc",
    ],
    alwayslink = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:all_kernels",
        ":cub",
#        "@cub_archive//:cub",
    ]
)
