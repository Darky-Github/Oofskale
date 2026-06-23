const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let session;


async function loadModel() {
  session = await ort.InferenceSession.create(
    "./Real-ESRGAN-General-x4v3.onnx",
    {
      executionProviders: ["webgpu", "wasm"]
    }
  );
}

loadModel();

document.getElementById("upload").onchange = async (e) => {
  const file = e.target.files[0];
  const img = new Image();

  img.onload = async () => {
    const w = img.width;
    const h = img.height;

    canvas.width = w;
    canvas.height = h;
    ctx.drawImage(img, 0, 0);

    const input = imageToTensor(img);

    const output = await session.run({ input: input });

    render(output);
  };

  img.src = URL.createObjectURL(file);
};

function imageToTensor(img) {
  const w = img.width;
  const h = img.height;

  ctx.drawImage(img, 0, 0);

  const imageData = ctx.getImageData(0, 0, w, h).data;

  const data = new Float32Array(w * h * 3);

  let j = 0;
  for (let i = 0; i < imageData.length; i += 4) {
    data[j++] = imageData[i] / 255;     // R
    data[j++] = imageData[i + 1] / 255; // G
    data[j++] = imageData[i + 2] / 255; // B
  }

  return new ort.Tensor("float32", data, [1, 3, h, w]);
}

function render(output) {
  const tensor = output[Object.keys(output)[0]];

  const [_, c, h, w] = tensor.dims;

  canvas.width = w;
  canvas.height = h;

  const imgData = ctx.createImageData(w, h);
  const data = tensor.data;

  let j = 0;
  for (let i = 0; i < w * h; i++) {
    imgData.data[i * 4] = data[j++] * 255;
    imgData.data[i * 4 + 1] = data[j++] * 255;
    imgData.data[i * 4 + 2] = data[j++] * 255;
    imgData.data[i * 4 + 3] = 255;
  }

  ctx.putImageData(imgData, 0, 0);
}
