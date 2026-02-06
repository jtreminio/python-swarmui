const apiBase = "/api";

async function apiRequest(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail || `Request failed (${response.status})`);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

function encodeImagePath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

function setBackendStatusText(snapshot) {
  const statusEl = document.getElementById("backend-status");
  if (!snapshot) {
    statusEl.textContent = "Backend: unknown";
    return;
  }
  const portText = snapshot.port ? `:${snapshot.port}` : "";
  const error = snapshot.last_error ? ` | ${snapshot.last_error}` : "";
  statusEl.textContent = `Backend: ${snapshot.status}${portText}${error}`;
}

function renderModelList(models) {
  const modelSelect = document.getElementById("model");
  const items = models["Stable-Diffusion"] || [];
  const priorValue = modelSelect.value;
  const preferredValue = modelSelect.dataset.userSelected || priorValue;
  modelSelect.innerHTML = "";
  if (!items.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No models found";
    modelSelect.appendChild(option);
    modelSelect.dataset.userSelected = "";
    return;
  }
  for (const model of items) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    modelSelect.appendChild(option);
  }

  if (preferredValue && items.includes(preferredValue)) {
    modelSelect.value = preferredValue;
    modelSelect.dataset.userSelected = preferredValue;
    return;
  }

  modelSelect.value = items[0];
  modelSelect.dataset.userSelected = items[0];
}

function renderJobs(jobs) {
  const container = document.getElementById("jobs");
  const meta = document.getElementById("jobs-meta");
  container.innerHTML = "";

  const active = jobs.filter((job) => job.status === "queued" || job.status === "running").length;
  meta.textContent = `${jobs.length} total job(s), ${active} active`;

  const template = document.getElementById("job-template");
  for (const job of jobs) {
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector(".job-id").textContent = job.id;
    node.querySelector(".job-status").textContent = job.status;
    node.querySelector(".job-message").textContent = job.error || job.message || "";
    node.querySelector(".progress").style.width = `${Math.max(0, Math.min(100, (job.progress || 0) * 100))}%`;

    const cancelButton = node.querySelector(".cancel");
    if (job.status === "queued" || job.status === "running") {
      cancelButton.onclick = async () => {
        try {
          await apiRequest(`/jobs/${job.id}/cancel`, { method: "POST" });
          await refreshJobs();
        } catch (error) {
          alert(`Cancel failed: ${error.message}`);
        }
      };
    } else {
      cancelButton.disabled = true;
    }

    container.appendChild(node);
  }
}

function renderGallery(images) {
  const container = document.getElementById("gallery");
  container.innerHTML = "";
  for (const image of images) {
    const item = document.createElement("article");
    item.className = "gallery-item";

    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `${apiBase}/image/${encodeImagePath(image.relative_path)}`;
    img.alt = image.file_name;

    const label = document.createElement("p");
    label.textContent = image.relative_path;

    item.appendChild(img);
    item.appendChild(label);
    container.appendChild(item);
  }
}

async function refreshStatus() {
  const payload = await apiRequest("/status");
  setBackendStatusText(payload.backend);
  renderModelList(payload.models || {});
}

async function refreshJobs() {
  const payload = await apiRequest("/jobs?limit=100");
  renderJobs(payload.jobs || []);
}

async function refreshGallery() {
  const payload = await apiRequest("/output");
  renderGallery(payload.images || []);
}

async function refreshAll() {
  await Promise.all([refreshStatus(), refreshJobs(), refreshGallery()]);
}

async function boot() {
  const modelSelect = document.getElementById("model");
  modelSelect.addEventListener("change", () => {
    modelSelect.dataset.userSelected = modelSelect.value;
  });

  document.getElementById("start-backend").onclick = async () => {
    try {
      await apiRequest("/backend/start", { method: "POST" });
      await refreshStatus();
    } catch (error) {
      alert(`Backend start failed: ${error.message}`);
    }
  };

  document.getElementById("restart-backend").onclick = async () => {
    try {
      await apiRequest("/backend/restart", { method: "POST" });
      await refreshStatus();
    } catch (error) {
      alert(`Backend restart failed: ${error.message}`);
    }
  };

  document.getElementById("stop-backend").onclick = async () => {
    try {
      await apiRequest("/backend/stop", { method: "POST" });
      await refreshStatus();
    } catch (error) {
      alert(`Backend stop failed: ${error.message}`);
    }
  };

  document.getElementById("generate-form").onsubmit = async (event) => {
    event.preventDefault();
    const body = {
      model: document.getElementById("model").value,
      prompt: document.getElementById("prompt").value,
      negative_prompt: document.getElementById("negative-prompt").value,
      steps: Number(document.getElementById("steps").value),
      cfg_scale: Number(document.getElementById("cfg").value),
      width: Number(document.getElementById("width").value),
      height: Number(document.getElementById("height").value),
      batch_size: Number(document.getElementById("batch-size").value),
      sampler_name: document.getElementById("sampler").value,
      scheduler: document.getElementById("scheduler").value,
      seed: Number(document.getElementById("seed").value),
    };
    if (!body.model) {
      alert("Select a model before generating.");
      return;
    }
    try {
      await apiRequest("/generate", {
        method: "POST",
        body: JSON.stringify(body),
      });
      await refreshJobs();
    } catch (error) {
      alert(`Generation request failed: ${error.message}`);
    }
  };

  await refreshAll();
  setInterval(refreshStatus, 5000);
  setInterval(refreshJobs, 2000);
  setInterval(refreshGallery, 4000);
}

boot().catch((error) => {
  alert(`Startup failed: ${error.message}`);
});
