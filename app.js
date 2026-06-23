const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const upload = document.getElementById("upload");
const btn = document.getElementById("upscaleBtn");
const status = document.getElementById("status");

let session;
let img = null;

async function loadModel() {
  status.innerText = "Loading model...";

  session = await ort.InferenceSession.create(
    "./Real-ESRGAN-General-x4v3.onnx",
    {
      executionProviders: ["webgpu", "wasm"]
    }
  );

  status.innerText = "Model ready";
}

loadModel();

upload.onchange = (e) => {
  const file = e.target.files[0];
  img = new Image();

  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    status.innerText = "Image loaded";
    btn.disabled = false;
  };

  img.src = URL.createObjectURL(file);
};

btn.onclick = async () => {
  if (!session || !img) return;

  btn.disabled = true;
  status.innerText = "Upscaling...";

  const input = imageToTensor(img);

  const output = await session.run({ input });

  render(output);

  status.innerText = "Done";
  btn.disabled = false;
};

function imageToTensor(img) {
  const w = img.width;
  const h = img.height;

  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, w, h).data;

  const tensorData = new Float32Array(w * h * 3);

  let j = 0;
  for (let i = 0; i < data.length; i += 4) {
    tensorData[j++] = data[i] / 255;
    tensorData[j++] = data[i + 1] / 255;
    tensorData[j++] = data[i + 2] / 255;
  }

  return new ort.Tensor("float32", tensorData, [1, 3, h, w]);
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
