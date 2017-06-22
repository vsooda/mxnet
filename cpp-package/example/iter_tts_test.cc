//
// Created by sooda on 17-6-21.
//

#include <iostream>
#include "mxnet-cpp/MxNetCpp.h"
// Allow IDE to parse the types
#include "../include/mxnet-cpp/op.h"

using namespace mxnet::cpp;

int main() {
  int batch_size = 64;
  std::string data_scp = "/home/sooda/speech/merlin/egs/world/s1/experiments/nana/duration_model/gen_data/lab.scp";
  std::string label_scp =  "/home/sooda/speech/merlin/egs/world/s1/experiments/nana/duration_model/gen_data/cmp.scp";
  auto train_iter = MXDataIter("TTSIter")
      .SetParam("data_scp", data_scp)
      .SetParam("label_scp", label_scp)
      .SetParam("data_shape", Shape(546))
      .SetParam("label_shape", Shape(5))
      .SetParam("batch_size", batch_size)
      .CreateDataIter();
  train_iter.Reset();
  int index = 0;
  while(train_iter.Next()) {
    auto data_batch = train_iter.GetDataBatch();
    std::cout << "============================iter: " << index++ << std::endl;
  }
  MXNotifyShutdown();
//  int a  = 0;
//  for (int i = 0; i < 10000000; i++) {
//    a++;
//  }
  std::cout << "hello" << std::endl;
}