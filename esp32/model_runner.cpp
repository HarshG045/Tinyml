// esp32/model_runner.cpp
#include "model_runner.h"

ModelRunner::ModelRunner(const unsigned char* model_data,
                         int model_len,
                         uint8_t* tensor_arena,
                         size_t arena_size) {
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch.");
    valid = false;
    return;
  }

  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, arena_size, nullptr);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed.");
    valid = false;
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  valid = true;
}

bool ModelRunner::invoke() {
  if (!valid) return false;
  TfLiteStatus status = interpreter->Invoke();
  return status == kTfLiteOk;
}
