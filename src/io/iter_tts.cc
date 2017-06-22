/*!
 *  Copyright (c) 2017 by sooda
 * \file iter_tts.cc
 * \brief define a TTS Reader to read in arrays
 */
#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/data.h>
#include "./iter_prefetcher.h"
#include "./iter_batchloader.h"
#include <fstream>
#include <string>
#include "../../dmlc-core/src/data/tts_parse.h"

namespace mxnet {
namespace io {

using dmlc::data::TTSParser;

// TTSIter parameters
struct TTSIterParam : public dmlc::Parameter<TTSIterParam> {
  /*! \brief path to data tts scp file */
  std::string data_scp;
  /*! \brief data shape */
  TShape data_shape;
  /*! \brief path to label tts scp file */
  std::string label_scp;
  /*! \brief label shape */
  TShape label_shape;
  // declare parameters
  DMLC_DECLARE_PARAMETER(TTSIterParam) {
    DMLC_DECLARE_FIELD(data_scp)
        .describe("The input TTS scp file");
    DMLC_DECLARE_FIELD(data_shape)
        .describe("The shape of one example.");
    DMLC_DECLARE_FIELD(label_scp).set_default("NULL")
        .describe("The input scp file. "
                  "If NULL, all labels will be returned as 0.");
    index_t shape1[] = {1};
    DMLC_DECLARE_FIELD(label_shape).set_default(TShape(shape1, shape1 + 1))
        .describe("The shape of one label.");
  }
};


class TTSIter: public IIterator<DataInst> {
 public:
  TTSIter() {
    out_.data.resize(2);
  }
  virtual ~TTSIter() {}

  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    data_parser_.reset(new TTSParser(param_.data_scp, param_.data_shape.Size()));
    if (param_.label_scp != "NULL") {
      label_parser_.reset(new TTSParser(param_.label_scp, param_.label_shape.Size()));
    } else {
      dummy_label.set_pad(false);
      dummy_label.Resize(mshadow::Shape1(1));
      dummy_label = 0.0f;
    }
  }

  virtual void BeforeFirst() {
    data_parser_->BeforeFirst();
    if (label_parser_.get() != nullptr) {
      label_parser_->BeforeFirst();
    }
    end_ = false;
    std::cout << "reset in beforefirst" << std::endl;
  }

  virtual bool Next() {
    if (end_) return false;

		if (!data_parser_->Next()) {
			end_ = true; return false;
		}
    out_.data[0] = AsTBlob(data_parser_->Value(), param_.data_shape);

    if (label_parser_.get() != nullptr) {
      if(!label_parser_->Next()){
        end_ = true; return false;
      }
      out_.data[1] = AsTBlob(label_parser_->Value(), param_.label_shape);
    } else {
      std::cout << "dummy:::::::::::::::" << std::endl;
      out_.data[1] = dummy_label;
    }
    return true;
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

 private:
  inline TBlob AsTBlob(const real_t* ptr, const TShape& shape) {
    return TBlob((real_t*)ptr, shape, cpu::kDevMask, 0);
  }

  TTSIterParam param_;
  // output instance
  DataInst out_;
  // at end
  bool end_{false};
  // dummy label
  mshadow::TensorContainer<cpu, 1, real_t> dummy_label;
  std::unique_ptr<TTSParser> label_parser_;
  std::unique_ptr<TTSParser> data_parser_;
};


DMLC_REGISTER_PARAMETER(TTSIterParam);

MXNET_REGISTER_IO_ITER(TTSIter)
.describe(R"code(Returns the TTS file iterator.
)code" ADD_FILELINE)
.add_arguments(TTSIterParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new TTSIter()));
  });

}  // namespace io
}  // namespace mxnet
