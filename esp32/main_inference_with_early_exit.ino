// esp32/main_inference_with_early_exit.ino

#include <Arduino.h>
#include "config.h"
#include "model_runner.h"
#include "sensor_capture.h"
#include "power_monitor.h"

// TFLite Micro
#include "TensorFlowLite.h"

// Models
#include "models/model_exit1_int8.h"
#include "models/model_full_int8.h"

// Tensor arenas
constexpr int kTensorArenaSize = 80 * 1024;
static uint8_t tensor_arena_exit1[kTensorArenaSize];
static uint8_t tensor_arena_full[kTensorArenaSize];

ModelRunner* runner_exit1 = nullptr;
ModelRunner* runner_full = nullptr;
PowerMonitor powerMonitor;

// Helper: dequantize for confidence
float dequantize(int8_t value, float scale, int zero_point) {
  return (static_cast<int>(value) - zero_point) * scale;
}

// Very simple argmax/confidence from int8 outputs
int argmax_and_confidence(TfLiteTensor* output, float& max_conf) {
  int num_classes = output->dims->data[1];  // assume [1, C]
  float scale = output->params.scale;
  int zero_point = output->params.zero_point;

  int max_idx = 0;
  float max_val = -1e9;

  for (int i = 0; i < num_classes; i++) {
    int8_t q = output->data.int8[i];
    float v = dequantize(q, scale, zero_point);
    if (v > max_val) {
      max_val = v;
      max_idx = i;
    }
  }

  max_conf = max_val;  // if logits, this is not exactly prob; still useful relatively
  return max_idx;
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("ESP32 TinyML Early-Exit Demo");

  // Power monitor
  if (powerMonitor.begin()) {
    Serial.println("INA219 initialized");
  } else {
    Serial.println("INA219 init failed or disabled");
  }

  // Model runners
  runner_exit1 = new ModelRunner(g_model_exit1_int8, g_model_exit1_int8_len,
                                 tensor_arena_exit1, kTensorArenaSize);
  runner_full = new ModelRunner(g_model_full_int8, g_model_full_int8_len,
                                tensor_arena_full, kTensorArenaSize);

  if (!runner_exit1->isValid() || !runner_full->isValid()) {
    Serial.println("ModelRunner init failed.");
    while (true) delay(1000);
  }

  Serial.println("Setup complete.");
}

void run_inference_cycle() {
  TfLiteTensor* in1 = runner_exit1->input();
  TfLiteTensor* out1 = runner_exit1->output();
  TfLiteTensor* in_full = runner_full->input();
  TfLiteTensor* out_full = runner_full->output();

  // 1) Capture & fill Exit1 input
  capture_and_preprocess_input(in1);

  // If shapes match, reuse same input for full model
  int input_len = IMG_H * IMG_W * IMG_C;
  for (int i = 0; i < input_len; i++) {
    in_full->data.int8[i] = in1->data.int8[i];
  }

  float v_before = powerMonitor.readVoltage();
  float i_before = powerMonitor.readCurrent_mA();

  unsigned long t_start = millis();
  bool ok1 = runner_exit1->invoke();
  unsigned long t_mid = millis();

  if (!ok1) {
    Serial.println("Exit1 model invoke failed.");
    return;
  }

  float conf1 = 0.0f;
  int class1 = argmax_and_confidence(out1, conf1);
  unsigned long t_exit1_ms = t_mid - t_start;

  bool early_exit_used = false;
  int final_class = -1;
  float final_conf = 0.0f;
  unsigned long total_time_ms = 0;

  if (conf1 >= EARLY_EXIT_THRESHOLD) {
    early_exit_used = true;
    final_class = class1;
    final_conf = conf1;
    total_time_ms = t_exit1_ms;
  } else {
    unsigned long t2_start = millis();
    bool ok_full = runner_full->invoke();
    unsigned long t2_end = millis();
    if (!ok_full) {
      Serial.println("Full model invoke failed.");
      return;
    }

    float conf2 = 0.0f;
    int class2 = argmax_and_confidence(out_full, conf2);
    final_class = class2;
    final_conf = conf2;
    total_time_ms = t2_end - t_start;
  }

  float v_after = powerMonitor.readVoltage();
  float i_after = powerMonitor.readCurrent_mA();

  float i_avg_mA = 0.5f * (i_before + i_after);
  float power_mW = v_before * i_avg_mA;
  float energy_mJ = power_mW * total_time_ms;  // rough relative metric

  Serial.println("---- Inference ----");
  Serial.print("Early exit: ");
  Serial.println(early_exit_used ? "YES" : "NO");
  Serial.print("Class: ");
  Serial.print(final_class);
  Serial.print(" | Conf (raw): ");
  Serial.println(final_conf, 4);
  Serial.print("Exit1 time (ms): ");
  Serial.println(t_exit1_ms);
  Serial.print("Total time (ms): ");
  Serial.println(total_time_ms);
  Serial.print("Approx power (mW): ");
  Serial.println(power_mW, 2);
  Serial.print("Approx energy (mJ, relative): ");
  Serial.println(energy_mJ, 2);
  Serial.println("-------------------");
}

void loop() {
  run_inference_cycle();
  delay(1000);  // 1 inference per second
}
