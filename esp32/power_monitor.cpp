// esp32/power_monitor.cpp
#include "power_monitor.h"

bool PowerMonitor::begin() {
#if ENABLE_POWER_MONITOR
  if (!ina219.begin()) {
    Serial.println("INA219 not found!");
    return false;
  }
  return true;
#else
  return false;
#endif
}

float PowerMonitor::readVoltage() {
#if ENABLE_POWER_MONITOR
  return ina219.getBusVoltage_V();
#else
  return 0.0f;
#endif
}

float PowerMonitor::readCurrent_mA() {
#if ENABLE_POWER_MONITOR
  return ina219.getCurrent_mA();
#else
  return 0.0f;
#endif
}
