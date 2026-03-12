const sampleButton = document.querySelector("#sample-button");
const inferButton = document.querySelector("#infer-button");
const samplePill = document.querySelector("#sample-pill");
const resultPill = document.querySelector("#result-pill");
const sourceFile = document.querySelector("#source-file");
const expectedLabel = document.querySelector("#expected-label");
const subjectLine = document.querySelector("#subject-line");
const urlCount = document.querySelector("#url-count");
const sampleText = document.querySelector("#sample-text");
const probabilityValue = document.querySelector("#probability-value");
const decisionLine = document.querySelector("#decision-line");
const predictedLabel = document.querySelector("#predicted-label");
const correctFlag = document.querySelector("#correct-flag");
const backendName = document.querySelector("#backend-name");
const urlRisk = document.querySelector("#url-risk");
const keywordList = document.querySelector("#keyword-list");
const statusLine = document.querySelector("#status-line");
const scoreRing = document.querySelector(".score-ring");

let currentSample = null;

function setStatus(message) {
  statusLine.textContent = message;
}

function renderKeywords(keywords) {
  keywordList.innerHTML = "";
  if (!keywords || keywords.length === 0) {
    const chip = document.createElement("span");
    chip.className = "keyword muted";
    chip.textContent = "No explanation keywords";
    keywordList.appendChild(chip);
    return;
  }
  keywords.forEach((keyword) => {
    const chip = document.createElement("span");
    chip.className = "keyword";
    chip.textContent = keyword;
    keywordList.appendChild(chip);
  });
}

function setProbability(probability) {
  const percentage = Math.round(probability * 100);
  probabilityValue.textContent = `${percentage}%`;
  const degrees = Math.max(0, Math.min(360, (probability || 0) * 360));
  scoreRing.style.background =
    `radial-gradient(circle at center, rgba(255, 252, 246, 0.98) 58%, transparent 60%), conic-gradient(var(--accent) ${degrees}deg, rgba(196, 77, 52, 0.16) ${degrees}deg)`;
}

function renderSample(sample) {
  currentSample = sample;
  inferButton.disabled = false;
  samplePill.textContent = `Sample #${sample.sample_id}`;
  sourceFile.textContent = sample.source_file || "-";
  expectedLabel.textContent = sample.label === 1 ? "phishing" : "benign";
  subjectLine.textContent = sample.subject || "(no subject)";
  urlCount.textContent = String(sample.urls.length);
  sampleText.textContent = sample.text;
  resultPill.className = "pill pill-neutral";
  resultPill.textContent = "Waiting";
  predictedLabel.textContent = "-";
  correctFlag.textContent = "-";
  backendName.textContent = "-";
  urlRisk.textContent = sample.urls.length ? "Available after inference" : "No URL in sample";
  decisionLine.textContent = "Run inference to see the model decision.";
  renderKeywords([]);
  setProbability(0);
}

async function loadRandomSample() {
  setStatus("Loading a random held-out sample...");
  const response = await fetch("/api/demo/random-sample");
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Unable to load a sample.");
  }
  renderSample(payload);
  setStatus("Sample loaded. Run inference when you are ready.");
}

async function runInference() {
  if (!currentSample) {
    return;
  }
  setStatus(`Running inference for sample #${currentSample.sample_id}...`);
  const response = await fetch("/api/demo/infer", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sample_id: currentSample.sample_id }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Unable to run inference.");
  }

  const prediction = payload.prediction;
  setProbability(prediction.phishing_probability);
  predictedLabel.textContent = prediction.predicted_label_name;
  correctFlag.textContent = prediction.is_correct ? "Yes" : "No";
  backendName.textContent = prediction.backend;
  decisionLine.textContent =
    `${prediction.predicted_label_name.toUpperCase()} at ${Math.round(prediction.phishing_probability * 100)}% confidence`;
  renderKeywords(prediction.explanation_keywords);

  if (prediction.is_correct) {
    resultPill.className = "pill pill-good";
    resultPill.textContent = "Correct";
  } else {
    resultPill.className = "pill pill-bad";
    resultPill.textContent = "Incorrect";
  }

  if (prediction.url_analysis) {
    urlRisk.textContent = `${Math.round(prediction.url_analysis.final_risk_score * 100)}% risk`;
  } else {
    urlRisk.textContent = "No URL analyzed";
  }

  setStatus(
    `Inference complete. The model predicted ${prediction.predicted_label_name} and it was ${prediction.is_correct ? "correct" : "incorrect"}.`
  );
}

sampleButton.addEventListener("click", () => {
  loadRandomSample().catch((error) => {
    setStatus(error.message);
  });
});

inferButton.addEventListener("click", () => {
  runInference().catch((error) => {
    setStatus(error.message);
  });
});
