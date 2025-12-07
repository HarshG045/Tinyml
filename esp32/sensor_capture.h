// esp32/sensor_capture.h
#pragma once

#include "TensorFlowLite.h"
#include "config.h"

// Fills the given input tensor with quantized data for model inference
void capture_and_preprocess_input(TfLiteTensor* input);
