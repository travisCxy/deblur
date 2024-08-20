#ifndef TENSORFLOW_CORE_KERNELS_ATTENTION_DECODER_H_
#define TENSORFLOW_CORE_KERNELS_ATTENTION_DECODER_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/kernels/rnn/blas_gemm.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/kernels/rnn/lstm_ops.h"

namespace tensorflow {
    namespace functor {
        template <typename Device, typename T, bool USE_CUBLAS, GateLayout gate_layout>
        struct AttentionDecoderBlockCellFprop_Part1 {
            void operator()(OpKernelContext* ctx, const Device& d,
                typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix a,
                typename TTypes<T, 3>::ConstTensor keys,
                typename TTypes<T>::ConstMatrix cs_prev,
                typename TTypes<T>::ConstMatrix h_prev,
                typename TTypes<T>::ConstMatrix lstm_w,
                typename TTypes<T>::ConstVec lstm_b,
                typename TTypes<T>::ConstMatrix query_w,
                typename TTypes<T>::ConstVec attention_v,
                typename TTypes<T>::Matrix xh,
                typename TTypes<T>::Matrix cs,
                typename TTypes<T>::Matrix gates,
                typename TTypes<T>::Matrix h,
                typename TTypes<T>::Matrix query,
                typename TTypes<T, 3>::Tensor score_base,
                typename TTypes<T, 2>::Tensor alignments,
                const int batch_size,
                const int x_size,
                const int a_size,
                const int h_size,
                const int time_len);
        };

        template<typename Device, typename T>
        struct AttentionDecoderBlockCellFprop_Part2 {
            void operator()(OpKernelContext* ctx, const Device& d,
                typename TTypes<T>::ConstMatrix h,
                typename TTypes<T>::ConstMatrix context,
                typename TTypes<T>::ConstMatrix attention_w,
                typename TTypes<T>::ConstMatrix project_w,
                typename TTypes<T>::ConstVec project_b,
                typename TTypes<T>::Matrix hc,
                typename TTypes<T>::Matrix attention,
                typename TTypes<T>::Matrix project_out,
                const int batch_size, const int h_size,
                const int context_size, const int project_size
                );
        };

        template <typename Device, typename T >
        struct GreedySamplerFprop {
            void operator()(OpKernelContext* ctx, const Device& d,
                typename TTypes<T>::ConstMatrix project_out,
                typename TTypes<bool>::ConstVec finished_inputs,
                typename TTypes<int>::Vec sample_ids,
                typename TTypes<T>::Matrix next_inputs,
                typename TTypes<bool>::Vec finished,
                const int start_token, const int end_token,
                const int batch_size, const int project_size
                );
        };

        template <typename Device, typename T >
        struct BeamSearchSamplerFprop {
            void operator()(OpKernelContext* ctx, const Device& d,
                typename TTypes<T>::Matrix project_out,
                typename TTypes<bool>::ConstVec finished_inputs,
                typename TTypes<T>::ConstVec log_probs_inputs,
                typename TTypes<T>::ConstMatrix h_status_inputs,
                typename TTypes<T>::ConstMatrix c_status_inputs,
                typename TTypes<T>::ConstMatrix a_status_inputs,

                typename TTypes<T>::Matrix h_status,
                typename TTypes<T>::Matrix c_status,
                typename TTypes<T>::Matrix a_status,

                typename TTypes<T>::Vec topk_values,
                typename TTypes<int>::Vec topk_indices,

                typename TTypes<int>::Vec beam_indices,
                typename TTypes<int>::Vec sample_ids,
                typename TTypes<T>::Matrix next_inputs,
                typename TTypes<bool>::Vec finished,
                const int beam_width,
                const int start_token, const int end_token,
                const int batch_size, const int project_size,
                const int cell_size
                );
        };

        // __global__ void beam_search_gather_status_test(T* dst_h_status, T* dst_c_status, T* dst_a_status, bool* dst_finished,
        //     const T* src_h_status, const T* src_c_status, const T* src_a_status, const bool * src_finished,
        //     const int* beam_indices, const int beam_width, const int batch_size, const int channel_size) {
        template <typename Device, typename T >
        struct BeamSearchBatchGatherFprop {
            void operator()(OpKernelContext* ctx, const Device& d,
                typename TTypes<T>::ConstMatrix src_h_status,
                typename TTypes<T>::ConstMatrix src_c_status,
                typename TTypes<T>::ConstMatrix src_a_status,
                typename TTypes<bool>::ConstVec src_finished,
                typename TTypes<int>::ConstVec beam_indices,
                typename TTypes<T>::Matrix dst_h_status,
                typename TTypes<T>::Matrix dst_c_status,
                typename TTypes<T>::Matrix dst_a_status,
                typename TTypes<bool>::Vec dst_finished,
                const int beam_width,
                const int batch_size,
                const int channel_size
                );
        };

        template <typename Device>
        struct CopyStepResult {
            void operator()(OpKernelContext* ctx, const Device& d,
                typename TTypes<int>::Vec step_result1,
                typename TTypes<int>::Vec step_result2,
                typename TTypes<int>::Matrix result1,
                typename TTypes<int>::Matrix result2,
                const int time,
                const int size,
                const int result_count
                );
        };


    };
}

#endif