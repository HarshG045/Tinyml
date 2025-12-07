// esp32/power_monitor.h
#pragma once

#include <Arduino.h>
#include <Adafruit_INA219.h>
#include "config.h"

class PowerMonitor {
public:
  bool begin();
  float readVoltage();    // V
  float readCurrent_mA(); // mA

private:
  Adafruit_INA219 ina219;
};
