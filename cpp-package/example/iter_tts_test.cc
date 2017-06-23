//
// Created by sooda on 17-6-21.
//

#include <iostream>
#include "mxnet-cpp/MxNetCpp.h"
// Allow IDE to parse the types
#include "../include/mxnet-cpp/op.h"

using namespace mxnet::cpp;

int main() {
  int batch_size = 32;
  std::string data_scp = "/home/sooda/speech/merlin/egs/world/s1/experiments/nana/duration_model/gen_data/lab.scp";
  std::string label_scp =  "/home/sooda/speech/merlin/egs/world/s1/experiments/nana/duration_model/gen_data/cmp.scp";
  auto train_iter = MXDataIter("TTSIter")
      .SetParam("data_scp", data_scp)
      .SetParam("label_scp", label_scp)
      .SetParam("data_shape", Shape(546))
      .SetParam("label_shape", Shape(5))
      .SetParam("batch_size", batch_size)
      .SetParam("round_batch", false)
      .CreateDataIter();
  train_iter.Reset(); //这个beforefirst的消息可能比下面next晚到达
  int index = 0;
  while(train_iter.Next()) {
    auto data_batch = train_iter.GetDataBatch();
    const mx_float* label = data_batch.label.GetData();
    const mx_float* data = data_batch.data.GetData();
    std::vector<mx_uint> label_shapes = data_batch.label.GetShape();
    std::vector<mx_uint> data_shapes = data_batch.data.GetShape();
    for (int i = 0; i < label_shapes[0]; i++) {
      for (int j = 0; j < label_shapes[1]; j++) {
        std::cout << label[i*label_shapes[1] + j] << " ";
      }
      std::cout << " --- ";
      for (int j = 0; j < label_shapes[1]; j++) {
        std::cout << data[i*data_shapes[1] + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "============================iter: " << index++ << std::endl;
  }

  MXNotifyShutdown();
//  int a  = 0;
//  for (int i = 0; i < 10000000; i++) {
//    a++;
//  }
  std::cout << "hello" << std::endl;
}