#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
    using shape_inference::DimensionHandle;
    using shape_inference::InferenceContext;
    using shape_inference::ShapeHandle;

    REGISTER_OP("AttentionBlockGreedyDecoderCell")
        .Input("keys: T")
        .Input("values: T")
        .Input("inputs: T")
        .Input("attention_inputs: T")
        .Input("lstm_cs_prev: T")
        .Input("lstm_h_prev: T")
        .Input("lstm_w: T")
        .Input("lstm_b: T")
        .Input("query_w: T")
        .Input("attention_v: T")
        .Input("attention_w: T")
        .Input("project_w: T")
        .Input("project_b: T")
        .Input("finished_inputs: bool")
        .Output("sample_ids: int32")
        .Output("finished: bool")
        .Output("next_inputs: T")
        .Output("project_out: T")
        .Output("next_lstm_cs: T")
        .Output("next_lstm_h: T")
        .Output("next_attention: T")
        .Output("alignments: T")
        .Attr("T: {half, float}")
        .Attr("start_token: int")
        .Attr("end_token: int")
        .SetShapeFn([](InferenceContext* c) {
        ShapeHandle keys, inputs, cs_prev;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &keys));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &inputs));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &cs_prev));

        DimensionHandle batch_size = c->Dim(inputs, 0);
        DimensionHandle output_size = c->Dim(inputs, 1);
        DimensionHandle max_time = c->Dim(keys, 1);
        DimensionHandle cell_size = c->Dim(cs_prev, 1);

        ShapeHandle sample_ids_shape = c->Vector(batch_size);
        c->set_output(0, sample_ids_shape); // sample_ids
        c->set_output(1, sample_ids_shape); // finished
        ShapeHandle next_inputs_shape = c->Matrix(batch_size, output_size);
        c->set_output(2, next_inputs_shape); // next_inputs
        c->set_output(3, next_inputs_shape); // project_out
        ShapeHandle status_shape = c->Matrix(batch_size, cell_size);
        c->set_output(4, status_shape); // next_lstm_cs
        c->set_output(5, status_shape); // next_lstm_h
        c->set_output(6, status_shape); // next_attention
        ShapeHandle alignments_shape = c->Matrix(batch_size, max_time);
        c->set_output(7, alignments_shape);
        return tensorflow::Status::OK();
    });

    REGISTER_OP("AttentionBlockBeamSearchDecoderCell")
        .Input("keys: T")
        .Input("values: T")
        .Input("inputs: T")
        .Input("attention_inputs: T")
        .Input("lstm_cs_prev: T")
        .Input("lstm_h_prev: T")
        .Input("lstm_w: T")
        .Input("lstm_b: T")
        .Input("query_w: T")
        .Input("attention_v: T")
        .Input("attention_w: T")
        .Input("project_w: T")
        .Input("project_b: T")
        .Input("finished_inputs: bool")
        .Input("log_probs_inputs: T")

        .Output("sample_ids: int32")
        .Output("finished: bool")
        .Output("next_inputs: T")
        .Output("project_out: T")
        .Output("next_lstm_cs: T")
        .Output("next_lstm_h: T")
        .Output("next_attention: T")
        .Output("beam_indices: int32")
        .Output("next_log_probs: T")
        .Output("topk_indices: int32")
        .Attr("T: {half, float}")
        .Attr("start_token: int")
        .Attr("end_token: int")
        .Attr("beam_width: int")
        .SetShapeFn([](InferenceContext* c) {
        ShapeHandle keys, inputs, cs_prev;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &keys));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &inputs));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &cs_prev));

        DimensionHandle batch_size = c->Dim(inputs, 0);
        DimensionHandle output_size = c->Dim(inputs, 1);
        DimensionHandle max_time = c->Dim(keys, 1);
        DimensionHandle cell_size = c->Dim(cs_prev, 1);

        ShapeHandle sample_ids_shape = c->Vector(batch_size);
        c->set_output(0, sample_ids_shape); // sample_ids
        c->set_output(1, sample_ids_shape); // finished
        ShapeHandle next_inputs_shape = c->Matrix(batch_size, output_size);
        c->set_output(2, next_inputs_shape); // next_inputs
        c->set_output(3, next_inputs_shape); // project_out
        ShapeHandle status_shape = c->Matrix(batch_size, cell_size);
        c->set_output(4, status_shape); // next_lstm_cs
        c->set_output(5, status_shape); // next_lstm_h
        c->set_output(6, status_shape); // next_attention
                                        //ShapeHandle alignments_shape = c->Matrix(batch_size, max_time);
                                        //c->set_output(7, alignments_shape);
        c->set_output(7, sample_ids_shape);
        c->set_output(8, sample_ids_shape);
        c->set_output(9, sample_ids_shape);
        return tensorflow::Status::OK();
    });


    REGISTER_OP("AttentionBlockGreedyDecoder")
        .Input("keys: T")
        .Input("values: T")
        .Input("inputs: T")
        .Input("attention_inputs: T")
        .Input("lstm_cs_prev: T")
        .Input("lstm_h_prev: T")
        .Input("lstm_w: T")
        .Input("lstm_b: T")
        .Input("query_w: T")
        .Input("attention_v: T")
        .Input("attention_w: T")
        .Input("project_w: T")
        .Input("project_b: T")
        .Input("finished_inputs: bool")
        .Output("sample_ids: int32")
        .Attr("T: {half, float}")
        .Attr("start_token: int")
        .Attr("end_token: int")
        .SetShapeFn([](InferenceContext* c) {
        ShapeHandle keys, inputs, cs_prev;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &keys));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &inputs));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &cs_prev));

        DimensionHandle batch_size = c->Dim(inputs, 0);
        DimensionHandle output_size = c->Dim(inputs, 1);
        DimensionHandle max_time = c->Dim(keys, 1);
        DimensionHandle cell_size = c->Dim(cs_prev, 1);

        ShapeHandle sample_ids_shape = c->Matrix(max_time, batch_size);
        c->set_output(0, sample_ids_shape); // sample_ids

        return tensorflow::Status::OK();
    });

    REGISTER_OP("AttentionBlockBeamSearchDecoder")
        .Input("keys: T")
        .Input("values: T")
        .Input("inputs: T")
        .Input("attention_inputs: T")
        .Input("lstm_cs_prev: T")
        .Input("lstm_h_prev: T")
        .Input("lstm_w: T")
        .Input("lstm_b: T")
        .Input("query_w: T")
        .Input("attention_v: T")
        .Input("attention_w: T")
        .Input("project_w: T")
        .Input("project_b: T")
        .Input("finished_inputs: bool")
        .Input("log_probs_inputs: T")

        .Output("sample_ids: int32")
        .Output("parent_ids: int32")
        .Attr("T: {half, float}")
        .Attr("start_token: int")
        .Attr("end_token: int")
        .Attr("beam_width: int")
        .SetShapeFn([](InferenceContext* c) {
        ShapeHandle keys, inputs, cs_prev;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &keys));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &inputs));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &cs_prev));

        DimensionHandle batch_size = c->Dim(inputs, 0);
        DimensionHandle output_size = c->Dim(inputs, 1);
        DimensionHandle max_time = c->Dim(keys, 1);
        DimensionHandle cell_size = c->Dim(cs_prev, 1);

        ShapeHandle sample_ids_shape = c->Matrix(max_time, batch_size);
        c->set_output(0, sample_ids_shape); // sample_ids
        c->set_output(1, sample_ids_shape); // finished

        return tensorflow::Status::OK();
    });

    REGISTER_OP("BeamSearchBatchGather")
        .Input("h_status: T")
        .Input("c_status: T")
        .Input("a_status: T")
        .Input("finished: bool")
        .Input("beam_indices: int32")
        .Output("h_status_out: T")
        .Output("c_status_out: T")
        .Output("a_status_out: T")
        .Output("finished_out: bool")
        .Attr("T: {half, float}")
        .Attr("beam_width: int")
        .SetShapeFn([](InferenceContext* c) {
        ShapeHandle h_status, finished;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &h_status));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &finished));

        c->set_output(0, h_status); // sample_ids
        c->set_output(1, h_status); // sample_ids
        c->set_output(2, h_status); // sample_ids
        c->set_output(3, finished); // sample_ids
        return tensorflow::Status::OK();
    });
}