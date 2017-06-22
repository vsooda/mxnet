//
// Created by sooda on 17-6-22.
//

#include <cstdlib>
#include <cstdio>
#include <dmlc/io.h>
#include <memory>
#include <iostream>
#include "../../dmlc-core/src/data/tts_parse.h"

using dmlc::data::TTSParser;

int main(int argc, char *argv[]) {
  std::string data_scp = "/home/sooda/speech/merlin/egs/world/s1/experiments/nana/duration_model/gen_data/cmp1.scp";
  int feat_dim = 5;
  std::unique_ptr<TTSParser> parser(new TTSParser(data_scp, feat_dim));
  parser->BeforeFirst();
  int line = 0;
  while (parser->Next()) {
    float* result = parser->Value();
    std::cout << line++  << " : ";
    for (int i = 0; i < feat_dim; i++) {
      std::cout << result[i] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
