/* Copyright 2015 Google Inc. All Rights Reserved.

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

// An example Op.
#define EIGEN_USE_THREADS

#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
vector<int> merge_repeated(const vector<int>& vec, int blank) {
    vector<int> ret;
    int last = -1;
    for (int i = 0; i < vec.size(); ++i) {
        if (vec[i] != last) {
            last = vec[i];
            if (vec[i] != blank) {
                ret.push_back(vec[i]);
            }
        }
    }
    return ret;
}
void get_top2(const float* data, int num_classes, int& idx1, int& idx2) {
    float v1 = -10000;
    idx1 = -1;
    float v2 = -10000;
    idx2 = -1;
    for (int i = 0; i < num_classes; ++i) {
        if (data[i] > v1) {
            v1 = data[i];
            idx1 = i;
        }
    }
    for (int i = 0; i < num_classes; ++i) {
        if (data[i] > v2 && i != idx1) {
            v2 = data[i];
            idx2 = i;
        }
    }
}

template<typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    // initialize original index locations
    vector < size_t > idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        idx[i] = i;
    }

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

    return idx;
}
string vec2code(const vector<int>& vec, const vector<string>& code_map) {
    string code = "";
    bool hw = false;
    for (int i = 0; i < vec.size(); ++i) {
        int v = vec[i];
        if (v == -1)
            break;
        if (v >= code_map.size()) {
            v -= code_map.size();
            if (!hw) {
                hw = true;
                code += "$";
            }
        }
        else if (hw) {
            hw = false;
            code += "$";
        }
        code += code_map[v];
    }
    if (hw) {
        code += "$";
    }
    return code;

}
struct seq_node {
    vector<int> indices;
    float acc_diff;
};
bool operator<(const seq_node& a, const seq_node& b) {
    return a.acc_diff < b.acc_diff;
}
int find_largest_index(const vector<seq_node>& vec_seq) {
    int p = 0;
    float v = 0;
    for (int i = 0; i < vec_seq.size(); ++i) {
        if (vec_seq[i].acc_diff > v) {
            p = i;
            v = vec_seq[i].acc_diff;
        }
    }
    return p;
}
int push_or_replace(vector<seq_node>& vec_seq, int largest_pos, int max_size,
    const seq_node& node) {
    if (vec_seq.size() < max_size) {
        vec_seq.push_back(node);
        return find_largest_index(vec_seq);
    }
    const seq_node& n1 = vec_seq[largest_pos];
    if (n1.acc_diff > node.acc_diff) {
        vec_seq[largest_pos] = node;
        return find_largest_index(vec_seq);
    }
    return largest_pos;

}
void print_debug(const vector<seq_node>& vec_seq, int largest_pos) {
    printf("begin %d\n", largest_pos);
    for (int i = 1; i < vec_seq.size(); ++i) {
        printf("\nseq node %d, total diff %.4f, ", i, vec_seq[i].acc_diff);
        for (int j = 0; j < vec_seq[i].indices.size(); ++j) {
            printf("%d ", vec_seq[i].indices[j]);
        }
    }
    printf("end\n");
}
vector<string> beam_decode_single(const float* vec, int seq_len,
    int num_classes, int k, const vector<string>& code_map,
    int& ref_start_ind, float& conf) {
    vector<string> ret;
    vector<float> diff(seq_len);
    vector<float> diff_conf(seq_len);
    vector<int> top1_indices(seq_len);
    vector<int> top2_indices(seq_len);
    conf = 0;
    ref_start_ind = k;
    int non_blank_cnt = 0;
    for (int i = 0; i < seq_len; ++i) {
        // get top2
        int idx1, idx2;
        get_top2(vec + i * num_classes, num_classes, idx1, idx2);
        diff[i] = vec[i * num_classes + idx1] - vec[i * num_classes + idx2];
        diff_conf[i] = diff[i] / vec[i * num_classes + idx1];
        top1_indices[i] = idx1;
        top2_indices[i] = idx2;
        if (idx1 != num_classes - 1) {
            conf += diff[i] / vec[i * num_classes + idx1];
            //printf("%d %d %d %.8f %.8f\n",i,idx1,idx2,vec[i * num_classes + idx1],vec[i * num_classes + idx2]);
            non_blank_cnt += 1;
        }
    }

    conf /= (non_blank_cnt + 1e-8);

    vector<int> top_indices = top1_indices;
    vector<int> top1_ret = merge_repeated(top_indices, num_classes - 1);
    string s = vec2code(top1_ret, code_map);
    ret.push_back(s);
    if (k == 1) {
        return ret;
    }

    vector <size_t> sorted_diff = sort_indexes<float>(diff);
    //for (int i = 0; i < seq_len; ++i) {
    //	printf("%d %.8f\n",(int)sorted_diff[i],diff[sorted_diff[i]]);
    //}
    //fflush(stdout);
    //vector<int> top_indices;

    float acc_diff = 0;
    vector<seq_node> vec_seq;
    seq_node zero_node;
    zero_node.acc_diff = 0;
    vec_seq.push_back(zero_node);
    vector<int> ans;
    int largest_pos = 0;
    for (int i = 0; i < seq_len; ++i) {
        int ind = sorted_diff[i];
        if (diff_conf[ind] > 0.3) {
            continue;
        }
        vector<seq_node> base_seq_node = vec_seq;
        for (int j = 0; j < base_seq_node.size(); ++j) {
            seq_node tmp_node = base_seq_node[j];
            tmp_node.indices.push_back(sorted_diff[i]);
            tmp_node.acc_diff += diff[sorted_diff[i]];
            largest_pos = push_or_replace(vec_seq, largest_pos, k * 3 + 1,
                tmp_node);
            //print_debug(vec_seq,largest_pos);
        }
        if (diff[sorted_diff[i]] > vec_seq[largest_pos].acc_diff) {
            break;
        }
        //if (i >= 5) break;
    }

    sort(vec_seq.begin(), vec_seq.end());
    float first_acc_diff = 0;
    for (int i = 1; i < vec_seq.size(); ++i) {
        if (ret.size() >= k) {
            break;
        }

        top_indices = top1_indices;
        //printf("seq node %d, total diff %.4f, ",i,vec_seq[i].acc_diff);
        //for (int j = 0; j < vec_seq[i].indices.size(); ++j) {
        //	printf("%d ",vec_seq[i].indices[j]);
        //}
        //printf("\n");
        for (int j = 0; j < vec_seq[i].indices.size(); ++j) {
            top_indices[vec_seq[i].indices[j]] =
                top2_indices[vec_seq[i].indices[j]];
        }
        ans = merge_repeated(top_indices, num_classes - 1);
        string s = vec2code(ans, code_map);
        int j = 0;
        for (; j < ret.size(); ++j) {
            if (s == ret[j]) {
                break;
            }
        }
        if (j == ret.size()) {
            ret.push_back(s);
        }

        if (i == 1) {
            first_acc_diff = vec_seq[i].acc_diff;
        }
        else if (ref_start_ind == k && vec_seq[i].acc_diff > first_acc_diff * 5) {
            ref_start_ind = ret.size();
        }

    }

    if (ref_start_ind == k) {
        ref_start_ind = ret.size();
    }
    return ret;
}
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector < std::string > elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

REGISTER_OP("CTCDecode")
.Input("inputs: float")
.Input("sequence_length: int32")
.Input("mapping: string")
.Attr("top_paths: int >= 1")
.Output("decoded_string: string")
.Output("ref_start_ind: int32")
.Output("confidence: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims));
    ::tensorflow::shape_inference::DimensionHandle batch_size;
    ::tensorflow::shape_inference::DimensionHandle max_time;
    batch_size = c->Dim(dims, 0);
    max_time = c->Dim(dims, 1);
    int32 top_paths;
    TF_RETURN_IF_ERROR(c->GetAttr("top_paths", &top_paths));
    ::tensorflow::shape_inference::ShapeHandle output_shape = c->MakeShape({ batch_size, top_paths });
    c->set_output(0, output_shape);
    ::tensorflow::shape_inference::ShapeHandle output_shape_2 = c->MakeShape({ batch_size });
    c->set_output(1, output_shape_2);
    c->set_output(2, output_shape_2);
    return ::tensorflow::Status::OK();
});

typedef Eigen::ThreadPoolDevice CPUDevice;

class CTCDecoderOp : public OpKernel {
public:
    explicit CTCDecoderOp(OpKernelConstruction* context) :
        OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("top_paths", &top_paths_));
    }

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& inputs = context->input(0);
        const Tensor& seq_lens = context->input(1);
        const Tensor& mapping = context->input(2);
        int batch_size = inputs.dim_size(0);
        int max_time = inputs.dim_size(1);
        int num_classes = inputs.dim_size(2);
        int vocab_size = mapping.dim_size(0);
        auto inputs_t = inputs.tensor<float, 3>();
        auto seq_len_t = seq_lens.vec<int32>();
        auto mapping_t = mapping.vec<tstring>();
        vector < string > code_map(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            code_map[i] = mapping_t(i);
            //fprintf(stderr, "%s\n", mapping_t(i).c_str());
        }

        Tensor* output_tensor = NULL;
        Tensor* output_tensor_2 = NULL;
        Tensor* output_tensor_3 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({
            batch_size, top_paths_ }), &output_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({
            batch_size }), &output_tensor_2));
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({
            batch_size }), &output_tensor_3));
        auto output_flat = output_tensor->flat<tstring>();
        auto output_flat_2 = output_tensor_2->flat<int>();
        auto output_flat_3 = output_tensor_3->flat<float>();

        auto shard_decode = [&](int begin, int end) {
            for (int i = begin; i < end; ++i) {
                int ref_start_ind = 0;
                float conf = 0;
                vector<string> single_ret = beam_decode_single(
                    inputs_t.data() + i * max_time * num_classes, seq_len_t(i),
                    num_classes, top_paths_, code_map, ref_start_ind, conf);
                for (int j = 0; j < single_ret.size(); ++j) {
                    output_flat(i * top_paths_ + j) = tstring(single_ret[j]);
                }
                output_flat_2(i) = ref_start_ind;
                output_flat_3(i) = conf;
            }
        };

        int input_bytes = batch_size * max_time * num_classes;
        int output_bytes = batch_size * max_time * num_classes * top_paths_;
        int compute_cycles = max_time * num_classes * top_paths_;
        const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
        const CPUDevice& d = context->eigen_device<CPUDevice>();
        d.parallelFor(batch_size, cost, shard_decode);

        /*
        for (int i = 0; i < batch_size; ++i) {
        int ref_start_ind = 0;
        float conf = 0;
        vector < string > single_ret = beam_decode_single(
        inputs_t.data() + i * max_time * num_classes, seq_len_t(i),
        num_classes, top_paths_, code_map, ref_start_ind, conf);
        for (int j = 0; j < single_ret.size(); ++j) {
        output_flat(i * top_paths_ + j) = tstring(single_ret[j]);
        }
        output_flat_2(i) = ref_start_ind;
        output_flat_3(i) = conf;
        }
        */
    }
private:
    int top_paths_;
};

REGISTER_KERNEL_BUILDER(Name("CTCDecode").Device(DEVICE_CPU), CTCDecoderOp);


REGISTER_OP("SampleIdDecode")
.Input("inputs: int32")
.Input("mapping: string")
.Attr("go_symbol: int")
.Attr("end_symbol: int")
.Attr("pad_symbol: int")
.Attr("offset: int")
.Output("decoded_string: string")

.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims));
    ::tensorflow::shape_inference::DimensionHandle batch_size;
    ::tensorflow::shape_inference::DimensionHandle beam_width;
    batch_size = c->Dim(dims, 0);
    beam_width = c->Dim(dims, 1);
    ::tensorflow::shape_inference::ShapeHandle output_shape = c->MakeShape({ batch_size, beam_width });
    c->set_output(0, output_shape);
    return ::tensorflow::Status::OK();
});

class SampleIdDecodeOp : public OpKernel {
public:
    explicit SampleIdDecodeOp(OpKernelConstruction* context) :
        OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("go_symbol", &go_symbol_));
        OP_REQUIRES_OK(context, context->GetAttr("end_symbol", &end_symbol_));
        OP_REQUIRES_OK(context, context->GetAttr("pad_symbol", &pad_symbol_));
        OP_REQUIRES_OK(context, context->GetAttr("offset", &offset_));
    }

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& inputs = context->input(0);
        const Tensor& mapping = context->input(1);
        int batch_size = inputs.dim_size(0);
        int beam_width = inputs.dim_size(1);
        int max_time = inputs.dim_size(2);


        int vocab_size = mapping.dim_size(0);
        auto inputs_t = inputs.tensor<int, 3>(); //inputs.flat<int>();

        auto mapping_t = mapping.vec<tstring>();
        vector < string > code_map(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            code_map[i] = mapping_t(i);
            //fprintf(stderr, "%s\n", mapping_t(i).c_str());
        }

        Tensor* output_tensor = NULL;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({
            batch_size, beam_width }), &output_tensor));

        auto output_flat = output_tensor->flat<tstring>();
        auto shard_decode = [&](int begin, int end) {
            for (int i = begin; i < end; i++) {
                for (int j = 0; j < beam_width; ++j) {
                    vector<int> vec;
                    for (int k = 0; k<max_time; ++k) {
                        int offset = i * beam_width * max_time + j * max_time + k;
                        //int ind = inputs_t.data()[offset];
                        int ind = inputs_t(offset);
                        if (ind == go_symbol_) {
                            continue;
                        }

                        if (ind == end_symbol_ || ind == pad_symbol_) {
                            break;
                        }

                        ind = ind - offset_;
                        vec.push_back(ind);
                    }
                    output_flat(i * beam_width + j) = tstring(vec2code(vec, code_map));
                }
            }
        };

        int input_bytes = batch_size * max_time * beam_width;
        int output_bytes = batch_size * max_time * beam_width;
        int compute_cycles = max_time * 100 * beam_width;
        const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
        const CPUDevice& d = context->eigen_device<CPUDevice>();
        d.parallelFor(batch_size, cost, shard_decode);

        //for (int i = 0; i < batch_size; ++i) {
        //    for (int j = 0; j < beam_width; ++j) {
        //        vector<int> vec;
        //        for (int k = 0; k<max_time; ++k) {
        //            int offset = i * beam_width * max_time + j * max_time + k;
        //            //int ind = inputs_t.data()[offset];
        //            int ind = inputs_t(offset);
        //            if (ind == go_symbol_) {
        //                continue;
        //            }
        //
        //            if (ind == end_symbol_ || ind == pad_symbol_) {
        //                break;
        //            }
        //
        //            ind = ind - offset_;
        //            vec.push_back(ind);
        //        }
        //        output_flat(i * beam_width + j) = tstring(vec2code(vec, code_map));
        //    }
        //}
    }
private:
    int go_symbol_;
    int end_symbol_;
    int pad_symbol_;
    int offset_;
};

REGISTER_KERNEL_BUILDER(Name("SampleIdDecode").Device(DEVICE_CPU), SampleIdDecodeOp);



REGISTER_OP("CTCGreedyDecode")
.Input("inputs: int32")
.Input("sequence_length: int32")
.Input("mapping: string")
.Attr("black: int")
.Output("decoded_string: string")

.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &dims));
    ::tensorflow::shape_inference::DimensionHandle batch_size;
    ::tensorflow::shape_inference::DimensionHandle max_time;
    batch_size = c->Dim(dims, 0);
    max_time = c->Dim(dims, 1);

    ::tensorflow::shape_inference::ShapeHandle output_shape = c->MakeShape({ batch_size });
    c->set_output(0, output_shape);
    return ::tensorflow::Status::OK();
});

class CtcGreedyDecoderOp : public OpKernel {
public:
    explicit CtcGreedyDecoderOp(OpKernelConstruction* context) :
        OpKernel(context) {

        OP_REQUIRES_OK(context, context->GetAttr("black", &black_ind_));
    }

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& inputs = context->input(0);
        const Tensor& seq_lens = context->input(1);
        const Tensor& mapping = context->input(2);
        int batch_size = inputs.dim_size(0);
        int max_time = inputs.dim_size(1);

        int vocab_size = mapping.dim_size(0);
        const int* inputs_t = inputs.flat<int>().data();  // inputs.tensor<int, 2>(); //inputs.flat<int>();

        auto seq_len_t = seq_lens.vec<int>();

        auto mapping_t = mapping.vec<tstring>();
        vector < string > code_map(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            code_map[i] = mapping_t(i);
            //fprintf(stderr, "%s\n", mapping_t(i).c_str());
        }

        Tensor* output_tensor = NULL;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({
            batch_size }), &output_tensor));

        auto output_flat = output_tensor->flat<tstring>();

        auto shard_decode = [&](int begin, int end) {
            for (int i = begin; i < end; i++) {
                vector<int> top_indices;
                top_indices.assign(inputs_t + i * max_time, inputs_t + i * max_time + min(max_time, seq_len_t(i)));
                vector<int> top1_ret = merge_repeated(top_indices, black_ind_);
                output_flat(i) = tstring(vec2code(top1_ret, code_map));
            }
        };

        int input_bytes = batch_size * max_time ;
        int output_bytes = batch_size * max_time;
        int compute_cycles = max_time * 100 ;
        const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
        const CPUDevice& d = context->eigen_device<CPUDevice>();
        d.parallelFor(batch_size, cost, shard_decode);
        //for (int i = 0; i < batch_size; ++i) {
        //    vector<int> top_indices;
        //    top_indices.assign(inputs_t + i * max_time, inputs_t + i * max_time + min(max_time, seq_len_t(i)));
        //    vector<int> top1_ret = merge_repeated(top_indices, black_ind_);
        //    output_flat(i) = tstring(vec2code(top1_ret, code_map));
        //}
    }
private:

    int black_ind_;
};

REGISTER_KERNEL_BUILDER(Name("CTCGreedyDecode").Device(DEVICE_CPU), CtcGreedyDecoderOp);