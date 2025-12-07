// esp32/config.h
#pragma once

// Image settings (must match training)
#define IMG_H 64
#define IMG_W 64
#define IMG_C 1  // 1 for grayscale, 3 for RGB

// Early exit
#define EARLY_EXIT_THRESHOLD 0.90f

// Feature flags
#define ENABLE_POWER_MONITOR 1

// INA219 shunt config (default Adafruit)
#define INA219_I2C_ADDR 0x40
