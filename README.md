# Real-ESRGAN Browser Upscaler (ONNX + WebGPU)

A fully client-side image upscaling tool that runs directly in the browser using ONNX Runtime Web and the Real-ESRGAN-General-x4v3 ONNX model.

No backend is required. All processing happens locally in the browser using WebGPU or WASM.

---

## Features

- Fully runs in the browser (no server required)
- Uses Real-ESRGAN General x4 upscaling model
- WebGPU acceleration with WASM fallback
- Compatible with GitHub Pages deployment
- No image uploads; all processing is local

---

## Project Structure

/index.html
/app.js
/Real-ESRGAN-General-x4v3.onnx

---

## Performance Notes

- WebGPU provides best performance on supported devices
- WASM fallback is slower but functional
- Large images may cause memory issues without tiling
- Recommended input size: up to 512–768px for stable performance

---

## Limitations

- No tiling support for large images
- No UI framework (intentionally minimal)
- WebGPU support depends on browser/device capability
- First load may be slow due to model size

---

## License

MIT

This project is for educational and experimental use only.

Real-ESRGAN model usage is subject to its original license.
