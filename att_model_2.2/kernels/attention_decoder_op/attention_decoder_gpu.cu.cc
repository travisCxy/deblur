/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
// #include "tensorflow/core/kernels/batch_matmul_op_impl.h"
#include "tensorflow/core/kernels/topk_op.h"
#include "attention_decoder.h"
#include "topk_gpu.h"
// #include "tensorflow/core/kernels/reduction_ops.h"
#include "reduction_gpu_kernels.cu.h"


namespace tensorflow {
    namespace functor {

        typedef Eigen::GpuDevice GPUDevice;

        namespace {

            struct FloatToHalf {
                __host__ __device__ EIGEN_STRONG_INLINE Eigen::half operator()(
                    const float& x) const {
                    return Eigen::half_impl::float_to_half_rtne(x);
                }
            };

            template <typename U, typename T>
            __host__ __device__ EIGEN_STRONG_INLINE
                typename std::enable_if<!std::is_same<T, U>::value, U>::type
                strict_cast(T t);

            template <typename U, typename T>
            __host__ __device__ EIGEN_STRONG_INLINE
                typename std::enable_if<std::is_same<T, U>::value, U>::type
                strict_cast(T t) {
                return t;
            }

            template <>
            __host__ __device__ EIGEN_STRONG_INLINE Eigen::half
                strict_cast<Eigen::half, float>(float t) {
                return FloatToHalf()(t);
            }

        }  // namespace

        template <typename T>
        struct TensorZero<GPUDevice, T> {
            void operator()(const GPUDevice& d, typename TTypes<T>::Flat t) {
                t.device(d) = t.constant(strict_cast<T>(0.f));
            }
        };

        template <typename T>
        struct TensorUnalignedZero<GPUDevice, T> {
            void operator()(const GPUDevice& d, typename TTypes<T>::UnalignedFlat t) {
                t.device(d) = t.constant(strict_cast<T>(0.f));
            }
        };

        namespace {

            // Adds bias, applies non-linearities and gates.
            //
            // Launch with a 2D setup such that there is one thread per (example,
            // activation) with 'x' governing example index and 'y' governing activation.
            //
            // Launch with blocks of (batch x 32)
            //
            // TODO(b/67600500): Try making 'use_peephole' a template parameter.
            template <typename T, GateLayout gate_layout>
            __global__ void lstm_gates(const T* gates, const T* b, const T* cs_prev,
                T* h, T* cs, const int batch_size, const int cell_size) {
                const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
                const int act_id = blockIdx.y * blockDim.y + threadIdx.y;



                if (batch_id >= batch_size || act_id >= cell_size) return;

                // The following code assumes the input arrays are of the following
                // shapes and interpretations.
                //
                // 1) 'gates' is a matrix such that,
                //
                //   cell_size  cell_size  cell_size  cell_size
                //  +----------+----------+----------+----------+
                //  |          |          |          |          |
                //  |    i     |    c     |    f     |    o     |  batch_size
                //  |          |          |          |          |
                //  +----------+----------+----------+----------+
                //
                // 'gid' is the index assigned to this thread for 'gates' in the 'i'
                // submatrix.
                //
                // 2) 'b' is a vector such that,
                //
                //   cell_size  cell_size  cell_size  cell_size
                //  +----------+----------+----------+----------+
                //  |    i     |    c     |    f     |    o     |  1
                //  +----------+----------+----------+----------+
                //
                // 'act_id' is the index assigned to this thread for 'b' in the 'i' subvector.
                //
                // 3) 'wc{i,f,o}' are vectors such that,
                //
                //   cell_size
                //  +----------+
                //  |    i     |  1
                //  +----------+
                //
                //  'act_id' is the index to this thread.
                //
                // 4) All other matrices have the form,
                //
                //   cell_size
                //  +----------+
                //  |          |
                //  |    i     |  batch_size
                //  |          |
                //  +----------+
                //
                // 'cid' is the index assigned to this thread.
                //
                const int gid = batch_id * cell_size * 4 + act_id;
                const int cid = batch_id * cell_size + act_id;
                Eigen::internal::scalar_logistic_op<T> sigmoid_op;
                Eigen::internal::scalar_tanh_op<T> tanh_op;
                Eigen::scalar_clip_op<T> clip_op;

                T i_local = sigmoid_op(gates[0 * cell_size + gid] + b[0 * cell_size + act_id]);

                const int c_offset = gate_c_offset(gate_layout, cell_size);
                const int f_offset = gate_f_offset(gate_layout, cell_size);

                const T ci_local = tanh_op(gates[c_offset + gid] + b[c_offset + act_id]);

                T f_local = sigmoid_op(gates[f_offset + gid] + b[f_offset + act_id]);

                T cs_local = i_local * ci_local + f_local * cs_prev[cid];
                cs[cid] = cs_local;

                const T co_local = tanh_op(cs_local);

                T o_local = sigmoid_op(gates[3 * cell_size + gid] + b[3 * cell_size + act_id]);

                h[cid] = o_local * co_local;
            }

            // Concatenate 'x' and 'h' and copy their contents into 'xh'.
            template <typename T>
            __global__ void concat_xah(T* xah, const T* x, const T* a, const T* h,
                const int batch_size, const int x_size, const int a_size,
                const int  h_size) {
                // Assumes 'x', 'h', and 'xh' are of the following shape,
                //
                //   input_size  cell_size
                //  +----------+----------+
                //  |          |          |
                //  |    x     |    h     |  batch_size
                //  |          |          |
                //  +----------+----------+
                //
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                const int width = x_size + a_size + h_size;

                if (gid >= width * batch_size) return;

                const int output_row = gid / width;
                const int output_col = gid % width;

                if (output_col < x_size) {  // x
                    xah[gid] = x[output_row * x_size + output_col];
                }
                else if (output_col < x_size + a_size) {
                    xah[gid] = a[output_row * a_size + output_col - x_size];
                }
                else {  // h
                    xah[gid] = h[output_row * h_size + output_col - x_size - a_size];
                }
            }

            template <typename T>
            __global__ void bahdanau_attention_score_base(T* out, const T* key, const T* query,
                const T* v,
                const int batch_size, const int time_len, const int channel_size) {
                // out shape = batch_size * time_len * channel_size
                // key shape = batch_size * time_len * channel_size
                // query_shape = batch_size * channel_size
                // v shape = channel_size
                // out = v * tanh(key + query) in element wise
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= batch_size * time_len * channel_size) return;

                const int batch_ind = gid / (time_len * channel_size);
                //const int time_ind = gid / channel_size;
                const int channel_ind = gid % channel_size;

                Eigen::internal::scalar_tanh_op<T> tanh_op;

                const T key_val = key[gid];
                const T query_val = query[batch_ind * channel_size + channel_ind];


                out[gid] = v[channel_ind] * tanh_op(key_val + query_val);
            }

            template <typename T>
            __global__ void bahdanau_attention_reduce_sum(T* out, const T* in, const int rows, const int columns) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= rows) return;

                T sum = (T)(0);
                for (int i = 0; i < columns; i++) {
                    sum += in[gid * columns + i];
                }
                out[gid] = sum;
                //printf("%d: %f\n", gid, sum);
            }

            template <typename T>
            __global__ void bahdanau_attention_reduce_max(T* out, const T* in, const int rows, const int columns) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= rows) return;

                T max_val = in[gid * columns];
                for (int i = 1; i < columns; i++) {
                    T tmp = in[gid * columns + i];
                    max_val = max_val > tmp ? max_val : tmp;
                }
                out[gid] = max_val;
                //printf("%d:  %f\n", gid,  max_val);
                //if(out[gid] != max_val) {
                //    printf("2  %d: %f, %f\n", gid, out[gid], max_val);
                //}
            }

            template <typename T>
            __global__ void bahdanau_attention_softmax(T* inout, const int rows, const int columns, const bool inlog) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= rows) return;

                T max_val = inout[gid * columns];
                for (int i = 1; i < columns; i++) {
                    T tmp = inout[gid * columns + i];
                    max_val = max_val > tmp ? max_val : tmp;
                }
                //printf("%d: %f\n", gid, max_val);

                T sum = (T)(0);
                for (int i = 0; i < columns; i++) {
                    T tmp = inout[gid * columns + i];
                    inout[gid * columns + i] = exp(tmp - max_val);
                    sum += inout[gid * columns + i];
                    //printf("%d: %f\n", gid * columns + i, inout[gid * columns + i]);
                }
                //printf("%d: %f\n", gid, sum);
                for (int i = 0; i < columns; i++) {
                    if (!inlog) {
                        inout[gid * columns + i] = inout[gid * columns + i] / sum;
                    }
                    else {
                        inout[gid * columns + i] = log(inout[gid * columns + i]) - log(sum);
                    }
                }
            }

            template <typename T>
            __global__ void bahdanau_attention_softmax_sub_exp(T* inout, T* max, const int rows, const int columns) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= rows * columns) return;

                const int row_ind = gid / columns;

                inout[gid] = exp(inout[gid] - max[row_ind]);
                //printf("%d: %f\n", gid, inout[gid]);
            }

            template <typename T>
            __global__ void bahdanau_attention_softmax_div(T* inout, T* sum, const int rows, const int columns, const bool in_log) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= rows * columns) return;

                const int row_ind = gid / columns;
                if (!in_log) {
                    inout[gid] = inout[gid] / sum[row_ind];
                }
                else {
                    inout[gid] = log(inout[gid]) - log(sum[row_ind]);
                }
            }

            template <typename T>
            __global__ void bahdanau_attention_bias_add(T* inout, const T* bias, const int batch_size, const int channel_size) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= batch_size * channel_size) return;

                const int channel_ind = gid % channel_size;

                inout[gid] += bias[channel_ind];
            }

            template <typename T>
            __global__ void bahdanau_attention_greedy_next_input(T* embedings, bool *finished, int* sample_ids, const bool *finished_inputs,
                const int start_token, const int end_token, const int batch_size, const int channel_size) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= batch_size * channel_size) return;

                const int batch_ind = gid / channel_size;
                const int channel_ind = gid % channel_size;

                const int sample_id = sample_ids[batch_ind];
                embedings[gid] = sample_id == channel_ind ? (T)(1.0) : (T)(0.0);

                if (channel_ind == 0) {
                    finished[batch_ind] = finished_inputs[batch_ind] || sample_ids[batch_ind] == end_token;
                }
            }

            template <typename T>
            __global__ void beam_search_update_log_probs(T* log_probs, const bool *finished_inputs,
                const T* log_probs_input, const int end_token, const int batch_size, const int channel_size) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= batch_size * channel_size) return;

                const int batch_ind = gid / channel_size;
                const int channel_ind = gid % channel_size;
                T val = log_probs[gid];

                if (finished_inputs[batch_ind]) {
                    if (channel_ind == end_token) {
                        val = (T)(0.0);
                    }
                    else {
                        val = (T)(-1e10); // a small num.
                    }
                }

                log_probs[gid] = val + log_probs_input[batch_ind];
            }

            template <typename T>
            __global__ void beam_search_next_inputs(T* next_inputs, int* beam_indices, int* sample_ids, bool* finished,
                int* topk_indices, const bool *finished_inputs, const int end_token, const int batch_size, const int channel_size) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= batch_size * channel_size) return;

                const int batch_ind = gid / channel_size;
                const int channel_ind = gid % channel_size;

                int sampled_id = topk_indices[batch_ind] % channel_size;
                if (channel_ind == 0) {
                    beam_indices[batch_ind] = topk_indices[batch_ind] / channel_size;
                    sample_ids[batch_ind] = sampled_id;
                    finished[batch_ind] = sampled_id == end_token;   // finished_inputs[batch_ind]
                }

                next_inputs[gid] = sampled_id == channel_ind ? (T)(1.0) : (T)(0.0);
            }

            template <typename T>
            __global__ void beam_search_gather_status(T* dst_h_status, T* dst_c_status, T* dst_a_status, bool* dst_finished,
                const T* src_h_status, const T* src_c_status, const T* src_a_status, const bool * src_finished,
                const int* beam_indices, const int beam_width, const int batch_size, const int channel_size, const bool dst_finished_has_value) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= batch_size * channel_size) return;

                const int batch_ind = gid / channel_size;
                const int channel_ind = gid % channel_size;

                const int true_batch_ind = batch_ind / beam_width;
                const int beam_ind = batch_ind % beam_width;

                const int select_beam_ind = beam_indices[batch_ind];


                const int select_gid = (true_batch_ind * beam_width + select_beam_ind) * channel_size + channel_ind;
                dst_h_status[gid] = src_h_status[select_gid];
                dst_c_status[gid] = src_c_status[select_gid];
                dst_a_status[gid] = src_a_status[select_gid];
                if (channel_ind == 0) {
                    if (dst_finished_has_value) {
                        dst_finished[batch_ind] = dst_finished[batch_ind] | src_finished[true_batch_ind * beam_width + select_beam_ind];
                    }
                    else {
                        dst_finished[batch_ind] = src_finished[true_batch_ind * beam_width + select_beam_ind];
                    }
                }
            }

            __global__ void decoder_copy_step_result(int* dst1, int* dst2, int* src1, int* src2, const int time, const int size,
                const int result_count) {
                const int gid = blockDim.x * blockIdx.x + threadIdx.x;
                if (gid >= size) return;

                const int dstInd = time * size + gid;
                dst1[dstInd] = src1[gid];
                if (result_count > 1) {
                    dst2[dstInd] = src2[gid];
                }
            }


            template <typename T>
            void DoSoftmax(OpKernelContext* ctx, const GPUDevice& d,
                typename TTypes<T>::Matrix logits,
                const bool is_log,
                const int rows,
                const int columns) {

                Tensor max_tensor;
                OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                    DataTypeToEnum<T>::v(),
                    TensorShape({ rows }),
                    &max_tensor));

                Tensor sum_tensor;
                OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                    DataTypeToEnum<T>::v(),
                    TensorShape({ rows }),
                    &sum_tensor));
                typename TTypes<T>::Vec max_values = max_tensor.vec<T>();

                typename TTypes<T>::Vec sum_values = sum_tensor.vec<T>();

                const auto& cu_stream = GetGpuStream(ctx);
                using MaxReducer = Eigen::internal::MaxReducer<T>;
                using Index = typename TTypes<T>::Tensor::Index;
                functor::ReduceFunctor<GPUDevice, MaxReducer>::Reduce(ctx, max_values, logits, Eigen::array<Index, 1>({ 1 }), MaxReducer());

                const int block_dim = 128;
                //const int grid_dim_0 = Eigen::divup(rows, block_dim);
                //TF_CHECK_OK(GpuLaunchKernel(bahdanau_attention_reduce_max<T>, grid_dim_0, block_dim, 0, cu_stream,
                //    max_values.data(), logits.data(), rows, columns));

                const int grid_dim = Eigen::divup(rows * columns, block_dim);
                TF_CHECK_OK(GpuLaunchKernel(bahdanau_attention_softmax_sub_exp<T>, grid_dim, block_dim, 0, cu_stream,
                    logits.data(), max_values.data(), rows, columns));

                using SumReducer = Eigen::internal::SumReducer<T>;
                functor::ReduceFunctor<GPUDevice, SumReducer>::Reduce(ctx, sum_values, logits, Eigen::array<Index, 1>({ 1 }), SumReducer());

                //TF_CHECK_OK(GpuLaunchKernel(bahdanau_attention_reduce_sum<T>, grid_dim_0, block_dim, 0, cu_stream,
                //    sum_values.data(), logits.data(), rows, columns));

                TF_CHECK_OK(GpuLaunchKernel(bahdanau_attention_softmax_div<T>, grid_dim, block_dim, 0, cu_stream,
                    logits.data(), sum_values.data(), rows, columns, is_log));
            }

            template <typename T, GateLayout gate_layout>
            void AttentionDecoderBlockCellFpropWithCUDA_PART1(
                OpKernelContext* ctx, const GPUDevice& d,
                typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix a,
                typename TTypes<T, 3>::ConstTensor keys,
                typename TTypes<T>::ConstMatrix cs_prev,
                typename TTypes<T>::ConstMatrix h_prev,
                typename TTypes<T>::ConstMatrix lstm_w,
                typename TTypes<T>::ConstVec lstm_b,
                typename TTypes<T>::ConstMatrix query_w,
                typename TTypes<T>::ConstVec attention_v,
                typename TTypes<T>::Matrix xah,
                typename TTypes<T>::Matrix cs,
                typename TTypes<T>::Matrix gates,
                typename TTypes<T>::Matrix h,
                typename TTypes<T>::Matrix query,
                typename TTypes<T, 3>::Tensor score_base,
                typename TTypes<T, 2>::Tensor alignments,
                const int batch_size, const int x_size,
                const int a_size, const int h_size,
                const int time_len) {
                const int cell_size = h_size;
                //const int input_size = x_size + a_size;
                const auto& cu_stream = GetGpuStream(ctx);

                // Concatenate xah = [x, a, h].
                //
                // Each block is assigned 128 threads. Good values are in [128, 1024] and are
                // divisible by 32 (the size of a warp). The number of blocks is such that
                // there are enough to process all the data.
                const int block_dim = 128;
                const int grid_dim =
                    Eigen::divup(batch_size * (x_size + a_size + h_size), block_dim);
                TF_CHECK_OK(GpuLaunchKernel(concat_xah<T>, grid_dim, block_dim, 0, cu_stream,
                    xah.data(), x.data(), a.data(), h_prev.data(), batch_size,
                    x_size, a_size, h_size));

                // states1 = xh * w
                typename TTypes<T>::ConstMatrix const_xah(xah.data(), xah.dimensions());
                TensorBlasGemm<GPUDevice, T, true /* USE_CUBLAS */>::compute(
                    ctx, d, false, false, typename gemm_compute_type<T>::type(1.f), const_xah,
                    lstm_w, typename gemm_compute_type<T>::type(0.f), gates);

                // Add bias, apply non-linearities and gating.
                //
                // Use 2D blocks. The number of threads per block is equal to x * y, where x =
                // min(batch_size, 8) and y = 32. See above for guidance on number of
                // threads.
                dim3 block_dim_2d(std::min(batch_size, 8), 32);
                dim3 grid_dim_2d(Eigen::divup(batch_size, static_cast<int>(block_dim_2d.x)),
                    Eigen::divup(cell_size, static_cast<int>(block_dim_2d.y)));


                TF_CHECK_OK(GpuLaunchKernel(
                    lstm_gates<T, gate_layout>, grid_dim_2d, block_dim_2d, 0,
                    cu_stream, gates.data(), lstm_b.data(), cs_prev.data(),
                    h.data(), cs.data(),
                    batch_size, cell_size));

                // query = cell_output
                typename TTypes<T>::ConstMatrix const_query(h.data(), h.dimensions());
                TensorBlasGemm<GPUDevice, T, true /* USE_CUBLAS */>::compute(
                    ctx, d, false, false, typename gemm_compute_type<T>::type(1.f), const_query,
                    query_w, typename gemm_compute_type<T>::type(0.f), query);

                // score_base
                const int grid_dim_2 = Eigen::divup(batch_size * time_len * cell_size, block_dim);
                TF_CHECK_OK(GpuLaunchKernel(bahdanau_attention_score_base<T>, grid_dim_2, block_dim, 0, cu_stream,
                    score_base.data(), keys.data(), query.data(), attention_v.data(), batch_size, time_len, cell_size));

                // query = cell_output
                typename TTypes<T>::ConstMatrix const_score_base(score_base.data(), { batch_size * time_len, cell_size });
                //typename TTypes<T>::ConstMatrix const_attention_v(attention_v.data(), { cell_size, 1 });
                typename TTypes<T>::Vec alignments_temp(alignments.data(), { batch_size * time_len });
                //TensorBlasGemm<GPUDevice, T, true /* USE_CUBLAS */>::compute(
                //    ctx, d, false, false, typename gemm_compute_type<T>::type(1.f), const_score_base,
                //    const_attention_v, typename gemm_compute_type<T>::type(0.f), alignments_temp);
                using Index = typename TTypes<T>::Tensor::Index;
                using SumReducer = Eigen::internal::SumReducer<T>;
                functor::ReduceFunctor<GPUDevice, SumReducer>::Reduce(ctx, alignments_temp, const_score_base, Eigen::array<Index, 1>({ 1 }), SumReducer());

                // score_base2
                const int grid_dim_4 = Eigen::divup(batch_size, block_dim);
                //TF_CHECK_OK(GpuLaunchKernel(bahdanau_attention_softmax<T>, grid_dim_4, block_dim, 0, cu_stream,
                //    alignments.data(), batch_size, time_len, false));
                DoSoftmax<T>(ctx, d, alignments, false, batch_size, time_len);
            }

            template <typename T >
            void AttentionDecoderBlockCellFpropWithCUDA_PART2(
                OpKernelContext* ctx, const GPUDevice& d,
                typename TTypes<T>::ConstMatrix h,
                typename TTypes<T>::ConstMatrix context,
                typename TTypes<T>::ConstMatrix attention_w,
                typename TTypes<T>::ConstMatrix project_w,
                typename TTypes<T>::ConstVec project_b,
                typename TTypes<T>::Matrix hc,
                typename TTypes<T>::Matrix attention,
                typename TTypes<T>::Matrix project_out,
                const int batch_size, const int h_size,
                const int context_size, const int project_size) {
                const auto& cu_stream = GetGpuStream(ctx);

                const int block_dim = 128;
                const int grid_dim = Eigen::divup(batch_size * (h_size + context_size), block_dim);
                TF_CHECK_OK(GpuLaunchKernel(concat_xah<T>, grid_dim, block_dim, 0, cu_stream,
                    hc.data(), h.data(), context.data(), context.data(), batch_size,
                    h_size, context_size, 0)); // fake ptr

                typename TTypes<T>::ConstMatrix const_hc(hc.data(), hc.dimensions());
                TensorBlasGemm<GPUDevice, T, true /* USE_CUBLAS */>::compute(
                    ctx, d, false, false, typename gemm_compute_type<T>::type(1.f), const_hc,
                    attention_w, typename gemm_compute_type<T>::type(0.f), attention);

                typename TTypes<T>::ConstMatrix const_attention(attention.data(), attention.dimensions());
                TensorBlasGemm<GPUDevice, T, true /* USE_CUBLAS */>::compute(
                    ctx, d, false, false, typename gemm_compute_type<T>::type(1.f), const_attention,
                    project_w, typename gemm_compute_type<T>::type(0.f), project_out);

                const int grid_dim2 = Eigen::divup(batch_size * project_size, block_dim);
                TF_CHECK_OK(GpuLaunchKernel(bahdanau_attention_bias_add<T>, grid_dim2, block_dim, 0, cu_stream,
                    project_out.data(), project_b.data(), batch_size, project_size)); // fake ptr
            }

            template <typename T >
            void GreedySamplerFpropWithCUDA(
                OpKernelContext* ctx, const GPUDevice& d,
                typename TTypes<T>::ConstMatrix project_out,
                typename TTypes<bool>::ConstVec finished_inputs,
                typename TTypes<int>::Vec sample_ids,
                typename TTypes<T>::Matrix next_inputs,
                typename TTypes<bool>::Vec finished,
                const int start_token, const int end_token,
                const int batch_size, const int project_size)
            {
                const auto& cu_stream = GetGpuStream(ctx);

                sample_ids.device(d) = project_out.argmax(1).template cast<int>();


                const int block_dim = 128;
                const int grid_dim = Eigen::divup(batch_size * project_size, block_dim);
                TF_CHECK_OK(GpuLaunchKernel(bahdanau_attention_greedy_next_input<T>, grid_dim, block_dim, 0, cu_stream,
                    next_inputs.data(), finished.data(), sample_ids.data(), finished_inputs.data(), start_token, end_token, batch_size,
                    project_size));
            }

            template <typename T >
            void BeamSearchSamplerFpropWithCUDA(
                OpKernelContext* ctx, const GPUDevice& d,
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
                const int cell_size)
            {
                const auto& cu_stream = GetGpuStream(ctx);

                // batch_size contains beam_width
                const int block_dim = 128;
                const int grid_dim = Eigen::divup(batch_size, block_dim);
                //TF_CHECK_OK(GpuLaunchKernel(bahdanau_attention_softmax<T>, grid_dim, block_dim, 0, cu_stream,
                //    project_out.data(), batch_size, project_size, true));
                DoSoftmax<T>(ctx, d, project_out, true, batch_size, project_size);

                //__global__ void beam_search_update_log_probs(T* log_probs, const bool *finished_inputs,
                //    T* log_probs_input, const int end_token, const int batch_size, const int channel_size) {

                const int grid_dim_2 = Eigen::divup(batch_size * project_size, block_dim);
                TF_CHECK_OK(GpuLaunchKernel(beam_search_update_log_probs<T>, grid_dim_2, block_dim, 0, cu_stream,
                    project_out.data(), finished_inputs.data(), log_probs_inputs.data(), end_token, batch_size, project_size));

                const int rows = batch_size / beam_width;
                const int columns = project_size * beam_width;
                typename TTypes<T>::ConstMatrix log_probs(project_out.data(), { rows, columns });
                typename TTypes<T>::Matrix topk_values_tmp(topk_values.data(), { rows, beam_width });
                typename TTypes<int>::Matrix topk_indices_tmp(topk_indices.data(), { rows, beam_width });
                TopKFunctor<GPUDevice, T>::Compute(ctx, true, beam_width, log_probs, rows, columns, topk_values_tmp, topk_indices_tmp);

                TF_CHECK_OK(GpuLaunchKernel(beam_search_next_inputs<T>, grid_dim_2, block_dim, 0, cu_stream,
                    next_inputs.data(), beam_indices.data(), sample_ids.data(), finished.data(),
                    topk_indices.data(), finished_inputs.data(), end_token, batch_size, project_size));

                const int grid_dim_3 = Eigen::divup(batch_size * cell_size, block_dim);
                TF_CHECK_OK(GpuLaunchKernel(beam_search_gather_status<T>, grid_dim_3, block_dim, 0, cu_stream,
                    h_status.data(), c_status.data(), a_status.data(), finished.data(),
                    h_status_inputs.data(), c_status_inputs.data(), a_status_inputs.data(), finished_inputs.data(),
                    beam_indices.data(), beam_width, batch_size, cell_size, true));
            }

            template <typename T >
            void BeamSearchBatchGatherFpropWithCUDA(
                OpKernelContext* ctx, const GPUDevice& d,
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
            ) {
                const auto& cu_stream = GetGpuStream(ctx);
                const int block_dim = 128;
                const int grid_dim = Eigen::divup(batch_size * channel_size, block_dim);
                TF_CHECK_OK(GpuLaunchKernel(beam_search_gather_status<T>, grid_dim, block_dim, 0, cu_stream,
                    dst_h_status.data(), dst_c_status.data(), dst_a_status.data(), dst_finished.data(),
                    src_h_status.data(), src_c_status.data(), src_a_status.data(), src_finished.data(),
                    beam_indices.data(), beam_width, batch_size, channel_size, false));
            }

            void CopyStepResultWithCUDA(
                OpKernelContext* ctx, const GPUDevice& d,
                typename TTypes<int>::Vec step_result1,
                typename TTypes<int>::Vec step_result2,
                typename TTypes<int>::Matrix result1,
                typename TTypes<int>::Matrix result2,
                const int time,
                const int size,
                const int result_count
            ) {
                const auto& cu_stream = GetGpuStream(ctx);
                const int block_dim = 128;
                const int grid_dim = Eigen::divup(size, block_dim);

                //__global__ void decoder_copy_step_result(T* dst1, T* dst2, T* src1, T* src2, const int time, const int size,
                //    const int result_count) {
                TF_CHECK_OK(GpuLaunchKernel(decoder_copy_step_result, grid_dim, block_dim, 0, cu_stream,
                    result1.data(), result2.data(), step_result1.data(), step_result2.data(),
                    time, size, result_count));
            }
        }  // namespace


        template <>
        void CopyStepResult<GPUDevice>::operator()(OpKernelContext* ctx, const GPUDevice& d,
                typename TTypes<int>::Vec step_result1,
                typename TTypes<int>::Vec step_result2,
                typename TTypes<int>::Matrix result1,
                typename TTypes<int>::Matrix result2,
                const int time,
                const int size,
                const int result_count
                ) {
            CopyStepResultWithCUDA(ctx, d, step_result1, step_result2,
                result1, result2, time, size, result_count);
        }
        template struct CopyStepResult<GPUDevice>;

#define DECLARE_GPU_FBPROP(T, GATE_LAYOUT)                                    \
  template <>                                                                 \
  void AttentionDecoderBlockCellFprop_Part1<GPUDevice, T, true /* USE_CUBLAS */, GATE_LAYOUT>:: \
  operator()(                                                                 \
      OpKernelContext* ctx, const GPUDevice& d,                               \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstMatrix a,                                      \
      typename TTypes<T, 3>::ConstTensor keys,                                \
      typename TTypes<T>::ConstMatrix cs_prev,                                \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix lstm_w,                                 \
      typename TTypes<T>::ConstVec lstm_b,                                    \
      typename TTypes<T>::ConstMatrix query_w,                                \
      typename TTypes<T>::ConstVec attention_v,                               \
      typename TTypes<T>::Matrix xh,                                          \
      typename TTypes<T>::Matrix cs,                                          \
      typename TTypes<T>::Matrix gates, typename TTypes<T>::Matrix h,         \
      typename TTypes<T>::Matrix query,                                      \
      typename TTypes<T, 3>::Tensor score_base,                              \
      typename TTypes<T, 2>::Tensor alignments,                              \
      const int batch_size, const int x_size, const int a_size,               \
      const int h_size, const int time_len                                                       \
        ) {                                                                   \
    AttentionDecoderBlockCellFpropWithCUDA_PART1<T, GATE_LAYOUT>(                   \
        ctx, d,  x, a, keys, cs_prev, h_prev, lstm_w,                                    \
        lstm_b, query_w, attention_v, xh, cs, gates,                               \
        h, query, score_base, alignments, batch_size,                               \
        x_size, a_size, h_size, time_len);                                          \
  }                                                                                 \
  template <>                                                                       \
  void AttentionDecoderBlockCellFprop_Part2<GPUDevice, T>::                         \
      operator()(                                                                   \
            OpKernelContext* ctx, const GPUDevice& d,                               \
            typename TTypes<T>::ConstMatrix h,                                      \
            typename TTypes<T>::ConstMatrix context,                                \
            typename TTypes<T>::ConstMatrix attention_w,                            \
            typename TTypes<T>::ConstMatrix project_w,                              \
            typename TTypes<T>::ConstVec project_b,                                 \
            typename TTypes<T>::Matrix hc,                                          \
            typename TTypes<T>::Matrix attention,                                   \
            typename TTypes<T>::Matrix project_out,                                 \
            const int batch_size, const int h_size,                                 \
            const int context_size, const int project_size                          \
            ) {                                                                     \
            AttentionDecoderBlockCellFpropWithCUDA_PART2<T>(                        \
                ctx, d, h, context, attention_w, project_w, project_b,              \
                hc, attention, project_out,                                         \
                batch_size, h_size, context_size, project_size);                    \
        }                                                                           \
  template <>                                                                       \
  void GreedySamplerFprop<GPUDevice, T>::                                           \
        operator()(OpKernelContext* ctx, const GPUDevice& d,                        \
            typename TTypes<T>::ConstMatrix project_out,                            \
            typename TTypes<bool>::ConstVec finished_inputs,                        \
            typename TTypes<int>::Vec sample_ids,                                 \
            typename TTypes<T>::Matrix next_inputs,                                 \
            typename TTypes<bool>::Vec finished,                                    \
            const int start_token, const int end_token,                         \
            const int batch_size, const int project_size                            \
            ) {                                                                     \
            GreedySamplerFpropWithCUDA<T>(ctx, d, project_out, finished_inputs,     \
                sample_ids,                                                         \
                next_inputs, finished, start_token, end_token, batch_size,          \
                project_size);                                                      \
        }                                                                           \
    template <>                                                                     \
    void BeamSearchSamplerFprop<GPUDevice, T>::                                     \
         operator()(OpKernelContext* ctx, const GPUDevice& d,                       \
            typename TTypes<T>::Matrix project_out,                                 \
            typename TTypes<bool>::ConstVec finished_inputs,                        \
            typename TTypes<T>::ConstVec log_probs_inputs,                          \
            typename TTypes<T>::ConstMatrix h_status_inputs,                            \
            typename TTypes<T>::ConstMatrix c_status_inputs,                               \
            typename TTypes<T>::ConstMatrix a_status_inputs,                               \
            typename TTypes<T>::Matrix h_status,                                    \
            typename TTypes<T>::Matrix c_status,                                    \
            typename TTypes<T>::Matrix a_status,                                    \
                                                                                    \
            typename TTypes<T>::Vec topk_values,                                 \
            typename TTypes<int>::Vec topk_indices,                              \
                                                                                    \
            typename TTypes<int>::Vec beam_indices,                                 \
            typename TTypes<int>::Vec sample_ids,                                   \
            typename TTypes<T>::Matrix next_inputs,                                 \
            typename TTypes<bool>::Vec finished,                                    \
            const int beam_width,                                                   \
            const int start_token, const int end_token,                             \
            const int batch_size, const int project_size,                            \
            const int cell_size                                                     \
            ){                                                                      \
        BeamSearchSamplerFpropWithCUDA<T>(                                             \
            ctx, d, project_out, finished_inputs, log_probs_inputs,                 \
            h_status_inputs, c_status_inputs, a_status_inputs,                      \
            h_status,                                                               \
            c_status, a_status, topk_values, topk_indices, beam_indices,            \
            sample_ids, next_inputs, finished, beam_width, start_token,             \
             end_token, batch_size,  project_size, cell_size);                                 \
        }                                                                           \
        template <>                                                                 \
        void BeamSearchBatchGatherFprop<GPUDevice, T>::                             \
            operator()(OpKernelContext* ctx, const GPUDevice& d,                    \
                typename TTypes<T>::ConstMatrix src_h_status,                       \
                typename TTypes<T>::ConstMatrix src_c_status,                       \
                typename TTypes<T>::ConstMatrix src_a_status,                       \
                typename TTypes<bool>::ConstVec src_finished,                          \
                typename TTypes<int>::ConstVec beam_indices,                     \
                typename TTypes<T>::Matrix dst_h_status,                            \
                typename TTypes<T>::Matrix dst_c_status,                            \
                typename TTypes<T>::Matrix dst_a_status,                            \
                typename TTypes<bool>::Vec dst_finished,                               \
                const int beam_width,                                               \
                const int batch_size,                                               \
                const int channel_size                                              \
                ) {                                                                 \
            BeamSearchBatchGatherFpropWithCUDA<T>(ctx, d, src_h_status,             \
                src_c_status, src_a_status, src_finished, beam_indices,             \
                dst_h_status, dst_c_status, dst_a_status, dst_finished,             \
                beam_width, batch_size, channel_size);                              \
        }                                                                           \
  template struct AttentionDecoderBlockCellFprop_Part1<GPUDevice, T, true /* USE_CUBLAS */,     \
                                     GATE_LAYOUT>;                                  \
  template struct AttentionDecoderBlockCellFprop_Part2<GPUDevice, T>;               \
  template struct GreedySamplerFprop<GPUDevice, T>;                                 \
  template struct BeamSearchSamplerFprop<GPUDevice, T>;


#define DECLARE_GPU_SPECS(T)                           \
  template struct TensorZero<GPUDevice, T>;            \
  template struct TensorUnalignedZero<GPUDevice, T>;   \
  template struct TensorCopy<GPUDevice, T>;            \
  template struct TensorCopyUnaligned<GPUDevice, T>;   \
  template struct TensorCopyToUnaligned<GPUDevice, T>; \
  template struct TensorAdd<GPUDevice, T>;             \
                                                       \
  DECLARE_GPU_FBPROP(T, IFCO);

        DECLARE_GPU_SPECS(Eigen::half);
        DECLARE_GPU_SPECS(float);
#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_FBPROP
    }  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
