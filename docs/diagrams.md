# System Architecture Diagrams

## 1. High-Level Workflow
[ Data Collection ] -> [ Python Training (Early Exit Model) ] -> [ Model Quantization (.tflite) ] -> [ C Array Conversion (.h) ] -> [ ESP32 Deployment ]

## 2. Early Exit Neural Network Architecture (Simplified)
Input -> Layer 1 -> Layer 2 -> [Exit 1 Branch] -> Layer 3 -> Layer 4 -> [Final Exit]

* If confidence at **Exit 1 Branch** >= Threshold, then Halt Inference.
* Else, continue to **Final Exit**.
