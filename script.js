let session = null;

const xMeans = [6.86504594, 0.27933767, 0.33273099, 6.45070189, 10.50884039];
const xScales = [0.84437537, 0.10159292, 0.11974234, 5.13865471, 1.22772995];

const yMean = 5.87136294;
const yScale = 0.88679941;

async function loadModel() {
  const resultText = document.getElementById("result");

  try {
    resultText.innerText = "Loading model";
    session = await ort.InferenceSession.create("wine_quality_model.onnx", {
  executionProviders: ['wasm']
});
    resultText.innerText = "Model loaded. Enter values and Predict.";
    console.log("Model loaded");
    console.log("Inputs:", session.inputNames);
    console.log("Outputs:", session.outputNames);
  } catch (error) {
    console.error("Model loading error:", error);
    resultText.innerText = "Error loading model: " + error.message;
  }
}

function scaleInput(values) {
  return values.map((v, i) => (v - xMeans[i]) / xScales[i]);
}

function unscaleOutput(value) {
  return value * yScale + yMean;
}

async function predictQuality() {
  const resultText = document.getElementById("result");

  if (!session) {
    resultText.innerText = "Model failed.";
    return;
  }

  const values = [
    parseFloat(document.getElementById("fixed_acidity").value),
    parseFloat(document.getElementById("volatile_acidity").value),
    parseFloat(document.getElementById("citric_acid").value),
    parseFloat(document.getElementById("residual_sugar").value),
    parseFloat(document.getElementById("alcohol").value)
  ];

  if (values.some(isNaN)) {
    resultText.innerText = "enter all values.";
    return;
  }

  try {
    const scaled = scaleInput(values);
    const tensor = new ort.Tensor("float32", Float32Array.from(scaled), [1, 5]);

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const results = await session.run({ [inputName]: tensor });
    const predictionScaled = results[outputName].data[0];
    const prediction = unscaleOutput(predictionScaled);

    resultText.innerText = "predicted wine quality: " + prediction.toFixed(2);
  } catch (error) {
    console.error("prediction error:", error);
    resultText.innerText = "prediction failed: " + error.message;
  }
}

loadModel();