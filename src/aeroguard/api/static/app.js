const state = {
  page: 1,
  pageSize: 8,
  risk: "",
  search: "",
  sortBy: "predicted_rul",
  order: "asc",
  selectedEngine: null,
  searchTimer: null,
  queueLoading: false,
};

const riskColors = {
  critical: "#ff6f73",
  high: "#f8ba4d",
  elevated: "#43e3ea",
  routine: "#62e6a7",
};
const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
let toastTimer;

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = String(value ?? "");
  return element.innerHTML;
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 25000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Request failed with status ${response.status}`);
    }
    return response.json();
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The request took too long. Please try again.");
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function showToast(message, type = "success") {
  const toast = document.querySelector("#toast");
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.className = `toast ${type} visible`;
  toastTimer = window.setTimeout(() => { toast.className = "toast"; }, 4200);
}

function setButtonLoading(button, loading, label) {
  button.disabled = loading;
  button.classList.toggle("loading", loading);
  if (label) button.querySelector(".button-label").textContent = label;
}

function setKpi(id, value) {
  const card = document.querySelector(id);
  card.querySelector(".kpi-value").textContent = value;
  card.classList.remove("skeleton-card");
}

function riskPill(band) {
  const safeBand = Object.hasOwn(riskColors, band) ? band : "elevated";
  return `<span class="pill ${safeBand}">${safeBand}</span>`;
}

function renderOverview(data) {
  const { fleet, model, retrieval } = data;
  setKpi("#kpi-engines", number.format(fleet.monitored_engines));
  setKpi("#kpi-critical", number.format(fleet.risk_counts.critical));
  setKpi("#kpi-rmse", `${number.format(model.random_forest_test_rmse)} cycles`);
  setKpi("#kpi-retrieval", `${number.format(retrieval.hit_rate_at_3 * 100)}%`);

  document.querySelector("#deployment-pill").textContent = `${model.deployment_champion.replaceAll("_", " ")} · deployed`;
  document.querySelector("#model-note").textContent =
    `LSTM is the validation accuracy winner, but promotion is ${model.lstm_promotion_status}. ` +
    "Random Forest stays deployed because its late-warning NASA score and direct SHAP auditability are stronger.";
  document.querySelector("#average-rul").textContent = `${number.format(fleet.average_deployment_rul)} avg RUL`;
  document.querySelector("#priority-message").textContent = `${fleet.risk_counts.critical} engines need critical review`;
  document.querySelector("#priority-detail").textContent =
    `Engine ${fleet.highest_priority_engine} leads the queue at ${number.format(fleet.lowest_predicted_rul)} predicted cycles remaining.`;

  const counts = fleet.risk_counts;
  const total = fleet.monitored_engines;
  const order = ["critical", "high", "elevated", "routine"];
  let cursor = 0;
  const slices = order.map((band) => {
    const start = cursor;
    cursor += total ? (counts[band] / total) * 100 : 0;
    return `${riskColors[band]} ${start}% ${cursor}%`;
  });
  const donut = document.querySelector("#risk-donut");
  donut.style.background = `conic-gradient(${slices.join(",")})`;
  donut.setAttribute("aria-label", order.map((band) => `${band} ${counts[band]}`).join(", "));
  document.querySelector("#donut-total").textContent = total;
  document.querySelector("#risk-legend").innerHTML = order.map((band) => `
    <div class="legend-item"><i style="background:${riskColors[band]}"></i><span>${band[0].toUpperCase() + band.slice(1)}</span><strong>${counts[band]}</strong></div>
  `).join("");
  return fleet.highest_priority_engine;
}

function renderModelBars(metrics) {
  const rows = metrics.filter((item) => item.evaluation === "official_test_endpoints_capped");
  const max = Math.max(...rows.map((item) => item.rmse), 1);
  document.querySelector("#model-bars").innerHTML = rows.map((item) => {
    const label = item.model === "lstm" ? "LSTM candidate" : "Random Forest";
    const secondary = item.model === "random_forest" ? "secondary" : "";
    return `<div class="model-row">
      <span class="model-name">${label}</span>
      <div class="bar-track" role="img" aria-label="${label} RMSE ${number.format(item.rmse)} cycles"><div class="bar-fill ${secondary}" data-width="${(item.rmse / max) * 100}"></div></div>
      <span class="model-score">${number.format(item.rmse)}</span>
    </div>`;
  }).join("");
  requestAnimationFrame(() => document.querySelectorAll(".bar-fill").forEach((bar) => {
    bar.style.width = `${bar.dataset.width}%`;
  }));
}

function renderRetrieval(data) {
  const summary = data.summary;
  document.querySelector("#metric-documents").textContent = summary.knowledge_documents;
  document.querySelector("#metric-chunks").textContent = summary.knowledge_chunks;
  document.querySelector("#metric-hit").textContent = `${number.format(summary.hit_rate_at_3 * 100)}%`;
  document.querySelector("#metric-mrr").textContent = summary.mean_reciprocal_rank.toFixed(3);
  document.querySelector("#knowledge-hash").textContent = `SHA-256 ${summary.knowledge_sha256.slice(0, 12)}…`;
}

function setFilter(risk) {
  state.risk = risk;
  state.page = 1;
  document.querySelectorAll(".filter").forEach((button) => {
    const selected = button.dataset.risk === risk;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function resetQueueControls() {
  state.search = "";
  state.sortBy = "predicted_rul";
  state.order = "asc";
  document.querySelector("#engine-search").value = "";
  document.querySelector("#clear-search").hidden = true;
  document.querySelector("#engine-sort").value = "predicted_rul:asc";
  setFilter("");
}

async function loadEngines() {
  if (state.queueLoading) return null;
  state.queueLoading = true;
  document.querySelector("#previous-page").disabled = true;
  document.querySelector("#next-page").disabled = true;
  const params = new URLSearchParams({
    page: state.page,
    page_size: state.pageSize,
    sort_by: state.sortBy,
    order: state.order,
  });
  if (state.risk) params.set("risk_band", state.risk);
  if (state.search) params.set("search", state.search);

  const table = document.querySelector("#engine-table");
  table.setAttribute("aria-busy", "true");
  let data;
  try {
    data = await api(`/api/engines?${params}`);
  } catch (error) {
    table.removeAttribute("aria-busy");
    state.queueLoading = false;
    throw error;
  }
  if (!data.items.length) {
    table.innerHTML = `<tr><td colspan="6" class="empty-cell"><div class="empty-state"><strong>No engines found</strong><span>Try another ID or reset the queue.</span><button type="button" class="reset-button" id="reset-queue">Reset filters</button></div></td></tr>`;
    document.querySelector("#reset-queue").addEventListener("click", async () => {
      resetQueueControls();
      await loadEngines();
    });
  } else {
    table.innerHTML = data.items.map((item) => `
      <tr tabindex="0" data-engine="${item.engine_id}" class="${state.selectedEngine === item.engine_id ? "selected" : ""}" aria-label="Open evidence for engine ${item.engine_id}">
        <td><strong>ENG-${String(item.engine_id).padStart(3, "0")}</strong></td>
        <td>${riskPill(item.risk_band)}</td>
        <td>${number.format(item.predicted_rul)}</td>
        <td>${number.format(item.advisory_lstm_rul)}</td>
        <td>${number.format(item.model_disagreement)}</td>
        <td><span class="row-arrow">›</span></td>
      </tr>
    `).join("");
    table.querySelectorAll("tr[data-engine]").forEach((row) => {
      const select = () => selectEngine(Number(row.dataset.engine));
      row.addEventListener("click", select);
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          select();
        }
      });
    });
  }
  table.removeAttribute("aria-busy");

  const pagination = data.pagination;
  document.querySelector("#queue-count").textContent = `${pagination.total_items} engines`;
  document.querySelector("#page-summary").textContent = `Page ${pagination.page} of ${pagination.total_pages} · ${pagination.total_items} results`;
  document.querySelector("#previous-page").disabled = pagination.page <= 1;
  document.querySelector("#next-page").disabled = pagination.page >= pagination.total_pages;
  state.queueLoading = false;
  return data;
}

function renderBrief(payload) {
  const brief = payload.brief;
  const evidence = payload.verified_evidence;
  const risk = document.querySelector("#brief-risk");
  risk.textContent = brief.risk_band;
  risk.className = `pill ${brief.risk_band}`;
  document.querySelector("#brief-content").innerHTML = `
    <p>${escapeHtml(brief.assessment)}</p>
    <div class="brief-metric">
      <div><span>Authoritative RF RUL</span><strong>${number.format(brief.predicted_rul_cycles)} cycles</strong></div>
      <div><span>Model disagreement</span><strong>${number.format(evidence.model_disagreement_cycles)} cycles</strong></div>
    </div>
    <h3>Human review actions</h3>
    <ul>${brief.recommended_actions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <h3>Retrieved citations</h3>
    <div class="citation-list">${brief.citations.map((item) => `<span class="citation">${escapeHtml(item)}</span>`).join("")}</div>
    <p class="panel-footnote">${escapeHtml(brief.limitations[0])}</p>`;
}

async function selectEngine(engineId, { scroll = false } = {}) {
  state.selectedEngine = engineId;
  document.querySelectorAll("tr[data-engine]").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.engine) === engineId);
  });
  document.querySelector("#brief-title").textContent = `Engine ${engineId} evidence`;
  const risk = document.querySelector("#brief-risk");
  risk.textContent = "Loading";
  risk.className = "pill";
  document.querySelector("#brief-content").innerHTML = `<div class="loading-line"></div><div class="loading-line short"></div>`;
  document.querySelector("#generate-brief").disabled = true;
  try {
    const payload = await api(`/api/engines/${engineId}/brief`);
    renderBrief(payload);
    document.querySelector("#generate-brief").disabled = false;
    if (scroll) document.querySelector("#copilot").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    document.querySelector("#brief-content").textContent = `Could not load this engine brief. ${error.message}`;
    showToast(error.message, "error");
  }
}

async function refreshAll({ announce = false } = {}) {
  const refreshButton = document.querySelector("#refresh-data");
  setButtonLoading(refreshButton, true, "Refreshing");
  try {
    const health = await api("/api/health");
    const status = document.querySelector("#system-state");
    status.className = `system-state ${health.status}`;
    status.querySelector("strong").textContent = health.ready ? "All evidence online" : "Artifacts incomplete";
    if (!health.ready) throw new Error("Run Days 1–4 to generate the required artifacts.");

    const [overview, metrics, retrieval] = await Promise.all([
      api("/api/overview"),
      api("/api/models"),
      api("/api/retrieval"),
    ]);
    const priorityEngine = renderOverview(overview);
    renderModelBars(metrics);
    renderRetrieval(retrieval);
    await loadEngines();
    await selectEngine(state.selectedEngine || priorityEngine);
    document.querySelector("#last-updated").textContent = `Updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    if (announce) showToast("Dashboard data refreshed.");
  } catch (error) {
    const status = document.querySelector("#system-state");
    status.className = "system-state degraded";
    status.querySelector("strong").textContent = "Attention required";
    showToast(error.message, "error");
  } finally {
    setButtonLoading(refreshButton, false, "Refresh data");
  }
}

document.querySelectorAll(".filter").forEach((button) => button.addEventListener("click", async () => {
  setFilter(button.dataset.risk);
  try { await loadEngines(); } catch (error) { showToast(error.message, "error"); }
}));

document.querySelector("#engine-search").addEventListener("input", (event) => {
  window.clearTimeout(state.searchTimer);
  const value = event.target.value.replace(/\D/g, "").slice(0, 6);
  event.target.value = value;
  document.querySelector("#clear-search").hidden = !value;
  state.searchTimer = window.setTimeout(async () => {
    state.search = value;
    state.page = 1;
    try { await loadEngines(); } catch (error) { showToast(error.message, "error"); }
  }, 250);
});

document.querySelector("#clear-search").addEventListener("click", async () => {
  document.querySelector("#engine-search").value = "";
  document.querySelector("#clear-search").hidden = true;
  state.search = "";
  state.page = 1;
  await loadEngines().catch((error) => showToast(error.message, "error"));
  document.querySelector("#engine-search").focus();
});

document.querySelector("#engine-sort").addEventListener("change", async (event) => {
  [state.sortBy, state.order] = event.target.value.split(":");
  state.page = 1;
  await loadEngines().catch((error) => showToast(error.message, "error"));
});

document.querySelector("#previous-page").addEventListener("click", async () => {
  if (state.page <= 1) return;
  state.page -= 1;
  await loadEngines().catch((error) => { state.page += 1; showToast(error.message, "error"); });
});

document.querySelector("#next-page").addEventListener("click", async () => {
  state.page += 1;
  await loadEngines().catch((error) => { state.page -= 1; showToast(error.message, "error"); });
});

document.querySelector("#review-critical").addEventListener("click", async () => {
  setFilter("critical");
  await loadEngines().catch((error) => showToast(error.message, "error"));
  document.querySelector("#fleet").scrollIntoView({ behavior: "smooth", block: "start" });
});

document.querySelector("#refresh-data").addEventListener("click", () => refreshAll({ announce: true }));

document.querySelector("#copilot-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.selectedEngine) return;
  const button = document.querySelector("#generate-brief");
  const question = document.querySelector("#copilot-question").value.trim();
  setButtonLoading(button, true, "Generating…");
  try {
    const payload = await api("/api/copilot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine_id: state.selectedEngine,
        question: question || null,
        generator_mode: "local",
      }),
    });
    renderBrief(payload);
    showToast(`Cited brief generated for engine ${state.selectedEngine}.`);
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    setButtonLoading(button, false, "Generate cited brief");
  }
});

document.querySelectorAll(".nav-item").forEach((link) => link.addEventListener("click", () => {
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
  link.classList.add("active");
}));

const observedSections = [...document.querySelectorAll("main section[id]")];
if ("IntersectionObserver" in window) {
  const sectionObserver = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.classList.toggle("active", item.dataset.section === visible.target.id);
    });
  }, { rootMargin: "-20% 0px -65%", threshold: [0.05, 0.35] });
  observedSections.forEach((section) => sectionObserver.observe(section));
}

refreshAll();
