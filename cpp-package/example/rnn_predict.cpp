#include <stdio.h>
#include <mxnet/c_predict_api.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            length_ = 0;
            buffer_ = NULL;
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        if (buffer_) {
          delete[] buffer_;
          buffer_ = NULL;
        }
    }
};

int main(int argc, char* argv[]) {

    // Models path for your model, you have to modify it
    std::string json_file = "roll.json";
    std::string param_file = "hh-0200.params";
    //std::string json_file = "ocr-symbol.json";
    //std::string param_file = "ocr-0006.params";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    int batch_size = 1;
    int seq_len = 1;

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    const mx_uint num_input_nodes = 2;
    const char* input_key[2] = {"data", "softmax_label"};
    const char** input_keys = input_key;
    const mx_uint  input_shape_indptr[3] = {0, 2, 4};
    const mx_uint input_shape_data[4] = {batch_size, seq_len, batch_size, seq_len};
//    const mx_uint num_input_nodes = 13;  // 1 for feedforward
//    const char* input_key[num_input_nodes] = {"data", "embed_weight",
//                                "lstm_l0_h2h_bias", "lstm_l0_h2h_weight",
//                                "lstm_l0_i2h_bias", "lstm_l0_i2h_weight",
//                                "lstm_l1_h2h_bias", "lstm_l1_h2h_weight",
//                                "lstm_l1_i2h_bias", "lstm_l1_i2h_weight",
//                                "pred_bias", "pred_weight", "label"};
//
//    const char** input_keys = input_key;
//
//    const mx_uint input_shape_indptr[num_input_nodes+1] = { 0, 2, 4, 6, 7, 9, 10, 12, 13, 15, 16, 17, 19, 21};
//    const mx_uint input_shape_data[21] = { 1, 10, 8713, 15,
//                                          60, 15, 60, 60, 15, 60,
//                                            60, 15, 60, 60, 15, 60,
//                                            8713, 8713, 15, 1, 10};

    PredictorHandle pred_hnd = 0;

    if (json_data.GetLength() == 0 ||
        param_data.GetLength() == 0) {
        return -1;
    }

    // Create Predictor
    MXPredCreate((const char*)json_data.GetBuffer(),
                 (const char*)param_data.GetBuffer(),
                 static_cast<size_t>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 &pred_hnd);
    assert(pred_hnd);

    std::vector<mx_float> input_data(seq_len);

    for (int i = 0; i < seq_len; i++) {
        input_data[i] = 1;
    }

    MXPredSetInput(pred_hnd, "data", input_data.data(), batch_size*seq_len);

    MXPredForward(pred_hnd);

    mx_uint output_index = 0;

    mx_uint *shape = 0;
    mx_uint shape_len;

    MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

    std::cout << size << std::endl;

    std::vector<float> data(size);

    MXPredGetOutput(pred_hnd, output_index, &(data[0]), size);

    // Release Predictor
    MXPredFree(pred_hnd);

    return 0;
}
