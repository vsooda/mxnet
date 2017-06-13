#include <stdio.h>
#include <mxnet/c_predict_api.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>

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


static std::tuple<std::unordered_map<wchar_t, mx_float>, std::vector<wchar_t>> loadCharIndices(
        const std::string file) {
    std::wifstream ifs(file, std::ios::binary);
    std::unordered_map<wchar_t, mx_float> map;
    std::vector<wchar_t> chars;
    if (ifs) {
        std::wostringstream os;
        os << ifs.rdbuf();
        int n = 1;
        map[L'\0'] = 0;
        chars.push_back(L'\0');
        for (auto c : os.str()) {
            map[c] = (mx_float) n++;
            chars.push_back(c);
        }
    }
    // Note: Can't use {} because this would hit the explicit constructor
    return std::tuple<std::unordered_map<wchar_t, mx_float>, std::vector<wchar_t>>(map, chars);
}

int main(int argc, char* argv[]) {

    //python symbol
    //std::string json_file = "roll_group.json";
    //std::string param_file = "hh-0200.params";

    //cpp symbol
    std::string json_file = "predict.json";
    std::string param_file = "mini-180.params";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    int batch_size = 1;
    int seq_len = 1;
    const int num_lstm_layer = 3;
    const int num_hidden = 512;
    const int state_shape_size = 2;



    const int add_input_num = num_lstm_layer * 2;
    const int add_shape_size = add_input_num * state_shape_size;

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    const mx_uint num_input_nodes = 8;
    const char* input_key[8] = {"data", "softmax_label",
                                "l0_init_c", "l0_init_h",
                                "l1_init_c", "l1_init_h",
                                "l2_init_c", "l2_init_h"};

    mx_uint  input_shape_indptr[3 + add_shape_size] = {0, 2, 4};
    mx_uint input_shape_data[4 + add_shape_size] = {batch_size, seq_len, batch_size, seq_len};

    int shape_ind_start_index = 3;
    int shape_data_start_index = 4;
    int name_start_index = 2;

//    std::vector<std::string> input_names;
//    input_names.push_back("data");
//    input_names.push_back("softmax_label");

    for (int i = 0; i < num_lstm_layer; i++) {
        std::string key = "l" + std::to_string(i) + "_init_";
        std::string key_c = key + "c";
        input_shape_indptr[shape_ind_start_index] = input_shape_indptr[shape_ind_start_index-1] + state_shape_size;
        shape_ind_start_index++;
        input_shape_data[shape_data_start_index++] = batch_size;
        input_shape_data[shape_data_start_index++] = num_hidden;
        //input_key[name_start_index++] = key_c.c_str();
        //input_names.push_back(key_c);

        std::string key_h = key + "h";
        input_shape_indptr[shape_ind_start_index] = input_shape_indptr[shape_ind_start_index-1] + state_shape_size;
        shape_ind_start_index++;
        input_shape_data[shape_data_start_index++] = batch_size;
        input_shape_data[shape_data_start_index++] = num_hidden;
        //input_names.push_back(key_h);
        //input_key[name_start_index++] = key_h.c_str();
    }

//    const char* input_key[num_input_nodes];
//    for (int i = 0; i < num_input_nodes; i++) {
//        input_key[i] = input_names[i].c_str();
//        std::cout << input_key[i] << std::endl;
//    }

    const char** input_keys = input_key;

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


    std::string dictionary_file = "mini.dictionary";
    auto dicts = loadCharIndices(dictionary_file);
    auto dictionary = std::get<0>(dicts);
    auto charIndices = std::get<1>(dicts);
    std::wstring text;

    bool seq_forward = true;
    if (seq_forward) {
        int start = 1;
        int seq_len = 100;
        int start_index = 66;
        int * presult = new int[seq_len];
        int softmax_dim = 0;
        MXPredSequnceForward(pred_hnd, seq_len, presult, softmax_dim, start_index);
        //output the result
        for (int i = 0; i < seq_len; i++) {
            int index = presult[i];
            if (index == 0) {
                break;
            }
            wchar_t c = charIndices[index];
            text.push_back(charIndices[index]);
        }
        std::wcout << text << std::endl;
    } else {
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
    }





    // Release Predictor
    MXPredFree(pred_hnd);

    return 0;
}
