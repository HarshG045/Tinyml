# Development Notes

## ESP32 Hardware
- Ensure the power monitor is correctly wired to measure current draw during inference.
- Using PlatformIO for easier library management with TFLite Micro.

## Model Training
- Int8 quantization significantly reduces model size and latency, allowing it to fit into ESP32 SRAM.
- The early exit tuner currently iterates over thresholds from 0.5 to 0.95 to find the optimal balance point.

## To-Do
- [ ] Add real power measurements to \power_results.csv\.
- [ ] Implement deeper neural network to benchmark more exit branches.
