// esp32/model_runner.h
#pragma once

#include <Arduino.h>
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

class ModelRunner {
public:
  ModelRunner(const unsigned char* model_data,
              int model_len,
              uint8_t* tensor_arena,
              size_t arena_size);

  bool isValid() const { return valid; }

  TfLiteTensor* input() { return interpreter->input(0); }
  TfLiteTensor* output() { return interpreter->output(0); }

  bool invoke();

private:
  const tflite::Model* model;
  tflite::AllOpsResolver resolver;
  tflite::MicroInterpreter* interpreter;
  TfLiteTensor* input_tensor;
  TfLiteTensor* output_tensor;

  bool valid = false;
};
