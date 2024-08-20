#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "attention_decoder.h"

#include <memory>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/kernels/batch_matmul_op_impl.h"
//#include "tensorflow/core/kernels/matmul_op_impl.h"

namespace tensorflow {

    typedef Eigen::ThreadPoolDevice CPUDevice;
    typedef Eigen::GpuDevice GPUDevice;

    template <typename Device, typename T, bool USE_CUBLAS, GateLayout gate_layout>
    void AttentionDecodeStep(OpKernelContext* ctx, const Device& device,
        const Tensor* keys_tensor, const Tensor* values_tensor,
        const Tensor* inputs_tensor, const Tensor* attention_inputs_tensor,
        const Tensor* cs_prev_tensor, const Tensor* h_prev_tensor,
        const Tensor* lstm_w_tensor, const Tensor* lstm_b_tensor,
        const Tensor* query_w_tensor, const Tensor* attention_v_tensor,
        const Tensor* attention_w_tensor, const Tensor* project_w_tensor,
        const Tensor* project_b_tensor,
        Tensor& xah_tensor, Tensor& cs_tensor,
        Tensor& gates_tensor, Tensor& h_tensor,
        Tensor& query_tensor, Tensor& score_base_tensor, Tensor& alignments_tensor,
        Tensor& context_tensor,
        Tensor& hc_tensor, Tensor& attention_tensor, Tensor& project_out_tensor,

        const int batch_size, const int x_size, const int a_size, const int cell_size,
        const int time_len, const int value_size, const int output_size
    ) {
        //LOG(ERROR) << "AAAAAAA";
        functor::AttentionDecoderBlockCellFprop_Part1<Device, T, USE_CUBLAS, gate_layout>()(
            ctx, device, inputs_tensor->matrix<T>(), attention_inputs_tensor->matrix<T>(),
            keys_tensor->tensor<T, 3>(), cs_prev_tensor->matrix<T>(),
            h_prev_tensor->matrix<T>(), lstm_w_tensor->matrix<T>(), lstm_b_tensor->vec<T>(),
            query_w_tensor->matrix<T>(), attention_v_tensor->vec<T>(),
            xah_tensor.matrix<T>(), cs_tensor.matrix<T>(),
            gates_tensor.matrix<T>(),
            h_tensor.matrix<T>(), query_tensor.matrix<T>(),
            score_base_tensor.tensor<T, 3>(),
            alignments_tensor.tensor<T, 2>(),
            batch_size, x_size, a_size, cell_size, time_len);
        //LOG(ERROR) << "bbbbbbb";
        Tensor expand_alignments_tensor;
        OP_REQUIRES(ctx,
            expand_alignments_tensor.CopyFrom(alignments_tensor, TensorShape({ batch_size, 1, time_len })),
            errors::Internal("Failed to reshape In[0] from ", alignments_tensor.shape().DebugString()));
        MatMulBCast bcast({ batch_size, 1, time_len }, { batch_size, time_len, value_size });
        //LaunchBatchMatMul<Device, T>::Launch(ctx, expand_alignments_tensor, *values_tensor, false, false, false, false, bcast, &context_tensor);
       LaunchBatchMatMul<Device, T>::Launch(ctx, expand_alignments_tensor, *values_tensor, false, false, false, false, bcast, &context_tensor);
        //LOG(ERROR) << "ccccccc";
        Tensor squeeze_context_tensor;
        OP_REQUIRES(ctx,
            squeeze_context_tensor.CopyFrom(context_tensor, TensorShape({ batch_size, value_size })),
            errors::Internal("Failed to reshape In[0] from ", context_tensor.shape().DebugString()));


        functor::AttentionDecoderBlockCellFprop_Part2<Device, T>()(
            ctx, device, const_cast<const Tensor*>(&h_tensor)->matrix<T>(),
            const_cast<const Tensor*>(&squeeze_context_tensor)->matrix<T>(),
            attention_w_tensor->matrix<T>(),
            project_w_tensor->matrix<T>(),
            project_b_tensor->vec<T>(),
            hc_tensor.matrix<T>(),
            attention_tensor.matrix<T>(),
            project_out_tensor.matrix<T>(),
            batch_size, cell_size, value_size,
            output_size
            );
        //    LOG(ERROR) << "ddddddd";
    }

    template <typename Device, typename T, bool USE_CUBLAS, GateLayout gate_layout>
    class AttentionBlockGreedyDecoderCellOp : public OpKernel {
    public:
        explicit AttentionBlockGreedyDecoderCellOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("start_token", &start_token_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("end_token", &end_token_));
        }

        void Compute(OpKernelContext* ctx) override {
            const Tensor* keys_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("keys", &keys_tensor));

            const Tensor* values_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));

            const Tensor* inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs_tensor));

            const Tensor* attention_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_inputs", &attention_inputs_tensor));

            const Tensor* cs_prev_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_cs_prev", &cs_prev_tensor));

            const Tensor* h_prev_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_h_prev", &h_prev_tensor));

            const Tensor* lstm_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_w", &lstm_w_tensor));

            const Tensor* lstm_b_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_b", &lstm_b_tensor));

            const Tensor* query_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("query_w", &query_w_tensor));

            const Tensor* attention_v_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_v", &attention_v_tensor));

            const Tensor* attention_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_w", &attention_w_tensor));

            const Tensor* project_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("project_w", &project_w_tensor));

            const Tensor* project_b_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("project_b", &project_b_tensor));

            const Tensor* finished_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("finished_inputs", &finished_inputs_tensor));

            const int64 batch_size = inputs_tensor->dim_size(0);
            const int64 x_size = inputs_tensor->dim_size(1);
            const int64 a_size = attention_inputs_tensor->dim_size(1);
            const int64 cell_size = cs_prev_tensor->dim_size(1);
            const int64 query_size = query_w_tensor->dim_size(1);
            const int64 time_len = keys_tensor->dim_size(1);
            const int64 value_size = values_tensor->dim_size(2);
            const int64 output_size = project_w_tensor->dim_size(1);

            // Sanity checks for our input shapes.
            OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                    cs_prev_tensor->dim_size(0), " vs. ",
                    batch_size));
            OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                    cs_prev_tensor->dim_size(1), " vs. ",
                    cell_size));

            OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                    h_prev_tensor->dim_size(0), " vs. ",
                    batch_size));
            OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));

            OP_REQUIRES(ctx, lstm_w_tensor->dim_size(0) == x_size + a_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != x_size + a_size + cell_size: ",
                    lstm_w_tensor->dim_size(0), " vs. ", x_size + a_size + cell_size));
            OP_REQUIRES(ctx, lstm_w_tensor->dim_size(1) == cell_size * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", lstm_w_tensor->dim_size(1),
                    " vs. ", cell_size * 4));

            OP_REQUIRES(ctx, lstm_b_tensor->dim_size(0) == cell_size * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", lstm_b_tensor->dim_size(0),
                    " vs. ", cell_size * 4));

            // Allocate our output tensors.
            Tensor* sample_ids_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("sample_ids", TensorShape({ batch_size }),
                    &sample_ids_tensor));

            Tensor* finished_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("finished", TensorShape({ batch_size }),
                    &finished_tensor));

            Tensor* next_inputs_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("next_inputs", TensorShape({ batch_size,  output_size }),
                    &next_inputs_tensor));

            Tensor* project_out_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("project_out", TensorShape({ batch_size,  output_size }),
                    &project_out_tensor));

            Tensor* cs_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("next_lstm_cs", TensorShape({ batch_size, cell_size }),
                    &cs_tensor));

            Tensor* h_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("next_lstm_h", TensorShape({ batch_size, cell_size }),
                    &h_tensor));

            Tensor* attention_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("next_attention", TensorShape({ batch_size,  cell_size }),
                    &attention_tensor));

            Tensor* alignments_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("alignments", TensorShape({ batch_size, time_len }),
                    &alignments_tensor));


            // Allocate our temp tensors.

            Tensor xah_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, x_size + a_size + cell_size }),
                &xah_tensor));

            Tensor gates_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size * 4 }),
                &gates_tensor));

            Tensor query_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size, query_size }),
                &query_tensor));

            Tensor score_base_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, time_len, query_size }),
                &score_base_tensor));

            Tensor context_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, 1,  value_size }),
                &context_tensor));

            Tensor hc_tensor;
            OP_REQUIRES_OK(ctx,
                ctx->allocate_temp(DataTypeToEnum<T>::v(),
                    TensorShape({ batch_size, cell_size + value_size }),
                    &hc_tensor));

            const Device& device = ctx->eigen_device<Device>();

            AttentionDecodeStep<Device, T, USE_CUBLAS, gate_layout>(ctx, device,
                keys_tensor, values_tensor,
                inputs_tensor, attention_inputs_tensor,
                cs_prev_tensor, h_prev_tensor,
                lstm_w_tensor, lstm_b_tensor,
                query_w_tensor, attention_v_tensor,
                attention_w_tensor, project_w_tensor,
                project_b_tensor,
                xah_tensor, *cs_tensor,
                gates_tensor, *h_tensor,
                query_tensor, score_base_tensor, *alignments_tensor,
                context_tensor,
                hc_tensor, *attention_tensor, *project_out_tensor,
                batch_size, x_size, a_size, cell_size,
                time_len, value_size, output_size
                );

            functor::GreedySamplerFprop<Device, T>()(
                ctx, device, const_cast<const Tensor*>(project_out_tensor)->matrix<T>(),
                finished_inputs_tensor->vec<bool>(),
                sample_ids_tensor->vec<int>(), next_inputs_tensor->matrix<T>(),
                finished_tensor->vec<bool>(),
                start_token_, end_token_,
                batch_size, output_size
                );
        }

    private:
        int start_token_;
        int end_token_;
    };

    template <typename Device, typename T, bool USE_CUBLAS, GateLayout gate_layout>
    class AttentionBlockBeamSearchDecoderCellOp : public OpKernel {
    public:
        explicit AttentionBlockBeamSearchDecoderCellOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("start_token", &start_token_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("end_token", &end_token_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
        }

        void Compute(OpKernelContext* ctx) override {
            const Tensor* keys_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("keys", &keys_tensor));

            const Tensor* values_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));

            const Tensor* inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs_tensor));

            const Tensor* attention_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_inputs", &attention_inputs_tensor));

            const Tensor* cs_prev_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_cs_prev", &cs_prev_tensor));

            const Tensor* h_prev_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_h_prev", &h_prev_tensor));

            const Tensor* lstm_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_w", &lstm_w_tensor));

            const Tensor* lstm_b_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_b", &lstm_b_tensor));

            const Tensor* query_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("query_w", &query_w_tensor));

            const Tensor* attention_v_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_v", &attention_v_tensor));

            const Tensor* attention_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_w", &attention_w_tensor));

            const Tensor* project_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("project_w", &project_w_tensor));

            const Tensor* project_b_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("project_b", &project_b_tensor));

            const Tensor* finished_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("finished_inputs", &finished_inputs_tensor));

            const Tensor* log_probs_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("log_probs_inputs", &log_probs_inputs_tensor));

            const int64 batch_size = inputs_tensor->dim_size(0);
            const int64 x_size = inputs_tensor->dim_size(1);
            const int64 a_size = attention_inputs_tensor->dim_size(1);
            const int64 cell_size = cs_prev_tensor->dim_size(1);
            const int64 query_size = query_w_tensor->dim_size(1);
            const int64 time_len = keys_tensor->dim_size(1);
            const int64 value_size = values_tensor->dim_size(2);
            const int64 output_size = project_w_tensor->dim_size(1);
            //LOG(ERROR) << "11111111111111111111111111";
            // Sanity checks for our input shapes.
            OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                    cs_prev_tensor->dim_size(0), " vs. ",
                    batch_size));
            OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                    cs_prev_tensor->dim_size(1), " vs. ",
                    cell_size));

            OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                    h_prev_tensor->dim_size(0), " vs. ",
                    batch_size));
            OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));

            OP_REQUIRES(ctx, lstm_w_tensor->dim_size(0) == x_size + a_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != x_size + a_size + cell_size: ",
                    lstm_w_tensor->dim_size(0), " vs. ", x_size + a_size + cell_size));
            OP_REQUIRES(ctx, lstm_w_tensor->dim_size(1) == cell_size * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", lstm_w_tensor->dim_size(1),
                    " vs. ", cell_size * 4));

            OP_REQUIRES(ctx, lstm_b_tensor->dim_size(0) == cell_size * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", lstm_b_tensor->dim_size(0),
                    " vs. ", cell_size * 4));

            // Allocate our output tensors.
            Tensor* sample_ids_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("sample_ids", TensorShape({ batch_size }),
                    &sample_ids_tensor));

            Tensor* finished_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("finished", TensorShape({ batch_size }),
                    &finished_tensor));

            Tensor* next_inputs_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("next_inputs", TensorShape({ batch_size,  output_size }),
                    &next_inputs_tensor));

            Tensor* project_out_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("project_out", TensorShape({ batch_size,  output_size }),
                    &project_out_tensor));

            Tensor* cs_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("next_lstm_cs", TensorShape({ batch_size, cell_size }),
                    &cs_tensor));

            Tensor* h_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("next_lstm_h", TensorShape({ batch_size, cell_size }),
                    &h_tensor));

            Tensor* attention_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("next_attention", TensorShape({ batch_size,  cell_size }),
                    &attention_tensor));
            //LOG(ERROR) << "ATTENTION TENSOR DTYPE: " << attention_tensor->dtype();

            Tensor* beam_indices_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("beam_indices", TensorShape({ batch_size }),
                    &beam_indices_tensor));

            Tensor *topk_indices_tensor;
            OP_REQUIRES_OK(ctx,
                ctx->allocate_output("topk_indices", TensorShape({ batch_size }),
                    &topk_indices_tensor));

            Tensor* topk_values_tensor;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("next_log_probs", TensorShape({ batch_size }),
                    &topk_values_tensor));



            // Allocate our temp tensors.

            Tensor xah_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, x_size + a_size + cell_size }),
                &xah_tensor));

            Tensor gates_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size * 4 }),
                &gates_tensor));

            Tensor query_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size, query_size }),
                &query_tensor));

            Tensor score_base_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, time_len, query_size }),
                &score_base_tensor));

            Tensor context_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, 1,  value_size }),
                &context_tensor));

            Tensor hc_tensor;
            OP_REQUIRES_OK(ctx,
                ctx->allocate_temp(DataTypeToEnum<T>::v(),
                    TensorShape({ batch_size, cell_size + value_size }),
                    &hc_tensor));

            Tensor alignments_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, time_len }),
                &alignments_tensor));


            Tensor cs_tmp_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size }),
                &cs_tmp_tensor));

            Tensor h_tmp_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size }),
                &h_tmp_tensor));

            Tensor attention_tmp_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size,  cell_size }),
                &attention_tmp_tensor));


            const Device& device = ctx->eigen_device<Device>();
            //LOG(ERROR) << "222222222222222222222222";
            AttentionDecodeStep<Device, T, USE_CUBLAS, gate_layout>(ctx, device,
                keys_tensor, values_tensor,
                inputs_tensor, attention_inputs_tensor,
                cs_prev_tensor, h_prev_tensor,
                lstm_w_tensor, lstm_b_tensor,
                query_w_tensor, attention_v_tensor,
                attention_w_tensor, project_w_tensor,
                project_b_tensor,
                xah_tensor, cs_tmp_tensor,
                gates_tensor, h_tmp_tensor,
                query_tensor, score_base_tensor, alignments_tensor,
                context_tensor,
                hc_tensor, attention_tmp_tensor, *project_out_tensor,
                batch_size, x_size, a_size, cell_size,
                time_len, value_size, output_size
                );


            functor::BeamSearchSamplerFprop<Device, T>()(
                ctx, device, project_out_tensor->matrix<T>(),
                finished_inputs_tensor->vec<bool>(),
                log_probs_inputs_tensor->vec<T>(),
                const_cast<const Tensor*>(&h_tmp_tensor)->matrix<T>(),
                const_cast<const Tensor*>(&cs_tmp_tensor)->matrix<T>(),
                const_cast<const Tensor*>(&attention_tmp_tensor)->matrix<T>(),

                h_tensor->matrix<T>(),
                cs_tensor->matrix<T>(),
                attention_tensor->matrix<T>(),

                topk_values_tensor->vec<T>(),
                topk_indices_tensor->vec<int>(),

                beam_indices_tensor->vec<int>(),
                sample_ids_tensor->vec<int>(),
                next_inputs_tensor->matrix<T>(),

                finished_tensor->vec<bool>(),
                beam_width_,
                start_token_, end_token_,
                (int)batch_size, (int)output_size,
                (int)cell_size
                );

        }

    private:
        int start_token_;
        int end_token_;
        int beam_width_;
    };




    template <typename Device, typename T, bool USE_CUBLAS, GateLayout gate_layout>
    class AttentionBlockGreedyDecoderOp : public OpKernel {
    public:
        explicit AttentionBlockGreedyDecoderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("start_token", &start_token_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("end_token", &end_token_));
        }

        void Compute(OpKernelContext* ctx) override {
            const Tensor* keys_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("keys", &keys_tensor));

            const Tensor* values_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));

            const Tensor* inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs_tensor));

            const Tensor* attention_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_inputs", &attention_inputs_tensor));

            const Tensor* cs_prev_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_cs_prev", &cs_prev_tensor));

            const Tensor* h_prev_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_h_prev", &h_prev_tensor));

            const Tensor* lstm_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_w", &lstm_w_tensor));

            const Tensor* lstm_b_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_b", &lstm_b_tensor));

            const Tensor* query_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("query_w", &query_w_tensor));

            const Tensor* attention_v_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_v", &attention_v_tensor));

            const Tensor* attention_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_w", &attention_w_tensor));

            const Tensor* project_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("project_w", &project_w_tensor));

            const Tensor* project_b_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("project_b", &project_b_tensor));

            const Tensor* finished_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("finished_inputs", &finished_inputs_tensor));

            const int64 batch_size = inputs_tensor->dim_size(0);
            const int64 x_size = inputs_tensor->dim_size(1);
            const int64 a_size = attention_inputs_tensor->dim_size(1);
            const int64 cell_size = cs_prev_tensor->dim_size(1);
            const int64 query_size = query_w_tensor->dim_size(1);
            const int64 time_len = keys_tensor->dim_size(1);
            const int64 value_size = values_tensor->dim_size(2);
            const int64 output_size = project_w_tensor->dim_size(1);

            // Sanity checks for our input shapes.
            OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                    cs_prev_tensor->dim_size(0), " vs. ",
                    batch_size));
            OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                    cs_prev_tensor->dim_size(1), " vs. ",
                    cell_size));

            OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                    h_prev_tensor->dim_size(0), " vs. ",
                    batch_size));
            OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));

            OP_REQUIRES(ctx, lstm_w_tensor->dim_size(0) == x_size + a_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != x_size + a_size + cell_size: ",
                    lstm_w_tensor->dim_size(0), " vs. ", x_size + a_size + cell_size));
            OP_REQUIRES(ctx, lstm_w_tensor->dim_size(1) == cell_size * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", lstm_w_tensor->dim_size(1),
                    " vs. ", cell_size * 4));

            OP_REQUIRES(ctx, lstm_b_tensor->dim_size(0) == cell_size * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", lstm_b_tensor->dim_size(0),
                    " vs. ", cell_size * 4));

            // Allocate our output tensors.
            Tensor* sample_ids_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("sample_ids", TensorShape({ time_len, batch_size }),
                    &sample_ids_tensor));

            // Allocate time_len batch
            Tensor finished_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<bool>::v(),
                TensorShape({ batch_size }),
                &finished_tensor));

            Tensor next_inputs_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size,  output_size }),
                &next_inputs_tensor));

            Tensor cs_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size }),
                &cs_tensor));

            Tensor h_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size }),
                &h_tensor));

            Tensor attention_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size,  cell_size }),
                &attention_tensor));

            // Allocate our temp tensors.
            Tensor project_out_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size,  output_size }),
                &project_out_tensor));

            Tensor alignments_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, time_len }),
                &alignments_tensor));


            Tensor xah_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, x_size + a_size + cell_size }),
                &xah_tensor));

            Tensor gates_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size * 4 }),
                &gates_tensor));

            Tensor query_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size, query_size }),
                &query_tensor));

            Tensor score_base_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, time_len, query_size }),
                &score_base_tensor));

            Tensor context_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, 1,  value_size }),
                &context_tensor));

            Tensor hc_tensor;
            OP_REQUIRES_OK(ctx,
                ctx->allocate_temp(DataTypeToEnum<T>::v(),
                    TensorShape({ batch_size, cell_size + value_size }),
                    &hc_tensor));

            Tensor step_sample_ids_tensor;
            OP_REQUIRES_OK(ctx,
                ctx->allocate_temp(DataTypeToEnum<T>::v(),
                    TensorShape({ batch_size }),
                    &step_sample_ids_tensor));

            const Device& device = ctx->eigen_device<Device>();
            //SliceHelper<Device, T> slicer(ctx);
            for (int t = 0; t < time_len; t++) {
                const Tensor *step_inputs_tensor = t == 0 ? inputs_tensor : const_cast<const Tensor*>(&next_inputs_tensor);
                const Tensor *step_attention_inputs_tensor = t == 0 ? attention_inputs_tensor : const_cast<const Tensor*>(&attention_tensor);
                const Tensor *step_cs_prev_tensor = t == 0 ? cs_prev_tensor : const_cast<const Tensor*>(&cs_tensor);
                const Tensor *step_h_prev_tensor = t == 0 ? h_prev_tensor : const_cast<const Tensor*>(&h_tensor);
                const Tensor *step_finished_inputs_tensor = t == 0 ? finished_inputs_tensor : const_cast<const Tensor*>(&finished_tensor);
                //Tensor step_sample_ids_tensor = slicer.OutputSlice(sample_ids_tensor, t, "sample_ids");
                //Tensor step_sample_ids_tensor = sample_ids_tensor->SubSlice(t);

                //LOG(ERROR) << "step_sample_ids_tensor is aligned : " << step_sample_ids_tensor.IsAligned() << step_sample_ids_tensor.shape();

                AttentionDecodeStep<Device, T, USE_CUBLAS, gate_layout>(ctx, device,
                    keys_tensor, values_tensor,
                    step_inputs_tensor, step_attention_inputs_tensor,
                    step_cs_prev_tensor, step_h_prev_tensor,
                    lstm_w_tensor, lstm_b_tensor,
                    query_w_tensor, attention_v_tensor,
                    attention_w_tensor, project_w_tensor,
                    project_b_tensor,
                    xah_tensor, cs_tensor,
                    gates_tensor, h_tensor,
                    query_tensor, score_base_tensor, alignments_tensor,
                    context_tensor,
                    hc_tensor, attention_tensor, project_out_tensor,
                    batch_size, x_size, a_size, cell_size,
                    time_len, value_size, output_size
                    );

                functor::GreedySamplerFprop<Device, T>()(
                    ctx, device, const_cast<const Tensor*>(&project_out_tensor)->matrix<T>(),
                    step_finished_inputs_tensor->vec<bool>(),
                    step_sample_ids_tensor.vec<int>(), next_inputs_tensor.matrix<T>(),
                    finished_tensor.vec<bool>(),
                    start_token_, end_token_,
                    batch_size, output_size
                    );

                functor::CopyStepResult<Device>()(
                    ctx, device, step_sample_ids_tensor.vec<int>(), step_sample_ids_tensor.vec<int>(),
                    sample_ids_tensor->matrix<int>(), sample_ids_tensor->matrix<int>(),
                    t, batch_size, 1
                    );
                //LOG(ERROR) << finished_tensor.vec<bool>().
                //LOG(ERROR) << "step_sample_ids_tensor is aligned2 : " << step_sample_ids_tensor.IsAligned();
                //slicer.FinishTimeStep();
            }
        }

    private:
        int start_token_;
        int end_token_;
    };

    template <typename Device, typename T, bool USE_CUBLAS, GateLayout gate_layout>
    class AttentionBlockBeamSearchDecoderOp : public OpKernel {
    public:
        explicit AttentionBlockBeamSearchDecoderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("start_token", &start_token_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("end_token", &end_token_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
        }

        void Compute(OpKernelContext* ctx) override {
            const Tensor* keys_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("keys", &keys_tensor));

            const Tensor* values_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));

            const Tensor* inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs_tensor));

            const Tensor* attention_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_inputs", &attention_inputs_tensor));

            const Tensor* cs_prev_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_cs_prev", &cs_prev_tensor));

            const Tensor* h_prev_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_h_prev", &h_prev_tensor));

            const Tensor* lstm_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_w", &lstm_w_tensor));

            const Tensor* lstm_b_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("lstm_b", &lstm_b_tensor));

            const Tensor* query_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("query_w", &query_w_tensor));

            const Tensor* attention_v_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_v", &attention_v_tensor));

            const Tensor* attention_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("attention_w", &attention_w_tensor));

            const Tensor* project_w_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("project_w", &project_w_tensor));

            const Tensor* project_b_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("project_b", &project_b_tensor));

            const Tensor* finished_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("finished_inputs", &finished_inputs_tensor));

            const Tensor* log_probs_inputs_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("log_probs_inputs", &log_probs_inputs_tensor));

            const int64 batch_size = inputs_tensor->dim_size(0);
            const int64 x_size = inputs_tensor->dim_size(1);
            const int64 a_size = attention_inputs_tensor->dim_size(1);
            const int64 cell_size = cs_prev_tensor->dim_size(1);
            const int64 query_size = query_w_tensor->dim_size(1);
            const int64 time_len = keys_tensor->dim_size(1);
            const int64 value_size = values_tensor->dim_size(2);
            const int64 output_size = project_w_tensor->dim_size(1);
            //LOG(ERROR) << "11111111111111111111111111";
            // Sanity checks for our input shapes.
            OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                    cs_prev_tensor->dim_size(0), " vs. ",
                    batch_size));
            OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                    cs_prev_tensor->dim_size(1), " vs. ",
                    cell_size));

            OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                    h_prev_tensor->dim_size(0), " vs. ",
                    batch_size));
            OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));

            OP_REQUIRES(ctx, lstm_w_tensor->dim_size(0) == x_size + a_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != x_size + a_size + cell_size: ",
                    lstm_w_tensor->dim_size(0), " vs. ", x_size + a_size + cell_size));
            OP_REQUIRES(ctx, lstm_w_tensor->dim_size(1) == cell_size * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", lstm_w_tensor->dim_size(1),
                    " vs. ", cell_size * 4));

            OP_REQUIRES(ctx, lstm_b_tensor->dim_size(0) == cell_size * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", lstm_b_tensor->dim_size(0),
                    " vs. ", cell_size * 4));

            // Allocate our output tensors.
            Tensor* sample_ids_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("sample_ids", TensorShape({ time_len, batch_size }),
                    &sample_ids_tensor));

            Tensor* parent_ids_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("parent_ids", TensorShape({ time_len, batch_size }),
                    &parent_ids_tensor));


            // Allocate our temp tensors.
            Tensor finished_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<bool>::v(), TensorShape({ batch_size }),
                    &finished_tensor));

            Tensor next_inputs_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size,  output_size }),
                    &next_inputs_tensor));

            Tensor project_out_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size,  output_size }),
                    &project_out_tensor));

            Tensor cs_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size, cell_size }),
                    &cs_tensor));

            Tensor h_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size, cell_size }),
                    &h_tensor));

            Tensor attention_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size,  cell_size }),
                    &attention_tensor));
            //LOG(ERROR) << "ATTENTION TENSOR DTYPE: " << attention_tensor->dtype();

            Tensor beam_indices_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size }),
                    &beam_indices_tensor));

            Tensor topk_indices_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<int>::v(), TensorShape({ batch_size }),
                    &topk_indices_tensor));

            Tensor topk_values_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size }),
                    &topk_values_tensor));


            Tensor xah_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, x_size + a_size + cell_size }),
                &xah_tensor));

            Tensor gates_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size * 4 }),
                &gates_tensor));

            Tensor query_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(), TensorShape({ batch_size, query_size }),
                &query_tensor));

            Tensor score_base_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, time_len, query_size }),
                &score_base_tensor));

            Tensor context_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, 1,  value_size }),
                &context_tensor));

            Tensor hc_tensor;
            OP_REQUIRES_OK(ctx,
                ctx->allocate_temp(DataTypeToEnum<T>::v(),
                    TensorShape({ batch_size, cell_size + value_size }),
                    &hc_tensor));

            Tensor alignments_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, time_len }),
                &alignments_tensor));


            Tensor cs_tmp_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size }),
                &cs_tmp_tensor));

            Tensor h_tmp_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size, cell_size }),
                &h_tmp_tensor));

            Tensor attention_tmp_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<T>::v(),
                TensorShape({ batch_size,  cell_size }),
                &attention_tmp_tensor));

            Tensor step_beam_indices_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<int>::v(),
                TensorShape({ batch_size}),
                &step_beam_indices_tensor));

            Tensor step_sample_ids_tensor;
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                DataTypeToEnum<int>::v(),
                TensorShape({ batch_size }),
                &step_sample_ids_tensor));

            const Device& device = ctx->eigen_device<Device>();
            //LOG(ERROR) << "222222222222222222222222";
            for (int t = 0; t < time_len; t++) {
                const Tensor *step_inputs_tensor = t == 0 ? inputs_tensor : const_cast<const Tensor*>(&next_inputs_tensor);
                const Tensor *step_attention_inputs_tensor = t == 0 ? attention_inputs_tensor : const_cast<const Tensor*>(&attention_tensor);
                const Tensor *step_cs_prev_tensor = t == 0 ? cs_prev_tensor : const_cast<const Tensor*>(&cs_tensor);
                const Tensor *step_h_prev_tensor = t == 0 ? h_prev_tensor : const_cast<const Tensor*>(&h_tensor);
                const Tensor *step_log_probs_inputs_tensor = t == 0 ? log_probs_inputs_tensor : const_cast<const Tensor*>(&topk_values_tensor);
                const Tensor *step_finished_inputs_tensor = t == 0 ? finished_inputs_tensor : const_cast<const Tensor*>(&finished_tensor);

                AttentionDecodeStep<Device, T, USE_CUBLAS, gate_layout>(ctx, device,
                    keys_tensor, values_tensor,
                    step_inputs_tensor, step_attention_inputs_tensor,
                    step_cs_prev_tensor, step_h_prev_tensor,
                    lstm_w_tensor, lstm_b_tensor,
                    query_w_tensor, attention_v_tensor,
                    attention_w_tensor, project_w_tensor,
                    project_b_tensor,
                    xah_tensor, cs_tmp_tensor,
                    gates_tensor, h_tmp_tensor,
                    query_tensor, score_base_tensor, alignments_tensor,
                    context_tensor,
                    hc_tensor, attention_tmp_tensor, project_out_tensor,
                    batch_size, x_size, a_size, cell_size,
                    time_len, value_size, output_size
                    );


                functor::BeamSearchSamplerFprop<Device, T>()(
                    ctx, device, project_out_tensor.matrix<T>(),
                    step_finished_inputs_tensor->vec<bool>(),
                    step_log_probs_inputs_tensor->vec<T>(),
                    const_cast<const Tensor*>(&h_tmp_tensor)->matrix<T>(),
                    const_cast<const Tensor*>(&cs_tmp_tensor)->matrix<T>(),
                    const_cast<const Tensor*>(&attention_tmp_tensor)->matrix<T>(),

                    h_tensor.matrix<T>(),
                    cs_tensor.matrix<T>(),
                    attention_tensor.matrix<T>(),

                    topk_values_tensor.vec<T>(),
                    topk_indices_tensor.vec<int>(),

                    step_beam_indices_tensor.vec<int>(),
                    step_sample_ids_tensor.vec<int>(),
                    next_inputs_tensor.matrix<T>(),

                    finished_tensor.vec<bool>(),
                    beam_width_,
                    start_token_, end_token_,
                    (int)batch_size, (int)output_size,
                    (int)cell_size
                    );

                functor::CopyStepResult<Device>()(
                    ctx, device, step_sample_ids_tensor.vec<int>(), step_beam_indices_tensor.vec<int>(),
                    sample_ids_tensor->matrix<int>(), parent_ids_tensor->matrix<int>(),
                    t, batch_size, 2
                    );
            }

        }

    private:
        int start_token_;
        int end_token_;
        int beam_width_;
    };



    template <typename Device, typename T>
    class BeamSearchBatchGatherOp : public OpKernel {
    public:
        explicit BeamSearchBatchGatherOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
        }

        void Compute(OpKernelContext* ctx) override {
            const Tensor* h_status_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("h_status", &h_status_tensor));

            const Tensor* c_status_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("c_status", &c_status_tensor));

            const Tensor* a_status_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("a_status", &a_status_tensor));

            const Tensor* finished_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("finished", &finished_tensor));

            const Tensor* beam_indices_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->input("beam_indices", &beam_indices_tensor));

            const int64 batch_size = h_status_tensor->dim_size(0);
            const int64 channel_size = h_status_tensor->dim_size(1);

            Tensor* h_status_out_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("h_status_out", TensorShape({ batch_size, channel_size }),
                    &h_status_out_tensor));

            Tensor* c_status_out_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("c_status_out", TensorShape({ batch_size, channel_size }),
                    &c_status_out_tensor));

            Tensor* a_status_out_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("a_status_out", TensorShape({ batch_size, channel_size }),
                    &a_status_out_tensor));

            Tensor* finished_out_tensor = nullptr;
            OP_REQUIRES_OK(
                ctx, ctx->allocate_output("finished_out", TensorShape({ batch_size, }),
                    &finished_out_tensor));

            const Device& device = ctx->eigen_device<Device>();
            functor::BeamSearchBatchGatherFprop<Device, T>()(
                ctx, device,
                h_status_tensor->matrix<T>(),
                c_status_tensor->matrix<T>(),
                a_status_tensor->matrix<T>(),
                finished_tensor->vec<bool>(),
                beam_indices_tensor->vec<int>(),
                h_status_out_tensor->matrix<T>(),
                c_status_out_tensor->matrix<T>(),
                a_status_out_tensor->matrix<T>(),
                finished_out_tensor->vec<bool>(),
                (int)beam_width_,
                (int)batch_size,
                (int)channel_size
                );
        }

    private:
        int beam_width_;
    };


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("AttentionBlockGreedyDecoderCell").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AttentionBlockGreedyDecoderCellOp<GPUDevice, T, true, IFCO>);                      \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("AttentionBlockBeamSearchDecoderCell").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AttentionBlockBeamSearchDecoderCellOp<GPUDevice, T, true, IFCO>);                      \
  REGISTER_KERNEL_BUILDER(                                                         \
        Name("AttentionBlockGreedyDecoder").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        AttentionBlockGreedyDecoderOp<GPUDevice, T, true, IFCO>);                     \
  REGISTER_KERNEL_BUILDER(                                                         \
        Name("AttentionBlockBeamSearchDecoder").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        AttentionBlockBeamSearchDecoderOp<GPUDevice, T, true, IFCO>);                     \
    REGISTER_KERNEL_BUILDER(                                                          \
        Name("BeamSearchBatchGather").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        BeamSearchBatchGatherOp<GPUDevice, T>);

    REGISTER_GPU_KERNEL(Eigen::half);
    REGISTER_GPU_KERNEL(float);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}
