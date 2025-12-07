// esp32/sensor_capture.cpp
#include "sensor_capture.h"
#include <Arduino.h>

// Dummy implementation: fills tensor with 0.5 intensity
void capture_and_preprocess_input(TfLiteTensor* input) {
  int input_len = IMG_H * IMG_W * IMG_C;
  float scale = input->params.scale;
  int zero_point = input->params.zero_point;

  float pixel_value_0_1 = 0.5f;  // mid-gray
  int8_t q = static_cast<int8_t>(round(pixel_value_0_1 / scale) + zero_point);

  for (int i = 0; i < input_len; i++) {
    input->data.int8[i] = q;
  }

  // TODO: replace with real sensor logic:
  // 1. Capture frame (camera or other)
  // 2. Resize to IMG_W x IMG_H
  // 3. Normalize to [0,1]
  // 4. Quantize using scale & zero_point
}
