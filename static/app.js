function el(id) {
  return document.getElementById(id);
}

let lastAssistantText = "";

function showToast(message) {
  let toast = document.getElementById("mini-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "mini-toast";
    toast.className = "mini-toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 1200);
}

function escapeHtml(text) {
  return (text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatMessage(text) {
  let out = escapeHtml(text);
  out = out.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\bP([0-3])\b/g, '<span class="prio-chip p$1">P$1</span>');
  out = out.replaceAll("\n", "<br>");
  return out;
}

function safeJson(obj) {
  return JSON.stringify(obj, null, 2);
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

function renderStage(stageListId, evt) {
  const ul = el(stageListId);
  if (!ul) return;

  const id = `stage-${evt.stage}`;
  let li = document.getElementById(id);
  if (!li) {
    li = document.createElement("li");
    li.id = id;
    ul.appendChild(li);
  }
  li.className = evt.status;
  li.textContent = `${evt.stage}: ${evt.status}`;
}

async function setupUpload() {
  const form = el("upload-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = el("file");
    if (!fileInput.files.length) return;

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);

    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    const out = el("upload-result");

    if (data.error) {
      out.innerHTML = `<div class="codebox">${data.error}</div>`;
      return;
    }

    const columns = data.columns || [];
    const mkCheckbox = (col, selected, kind) => {
      return `<label><input type="checkbox" data-kind="${kind}" value="${col}" ${selected ? "checked" : ""}/> ${col}</label>`;
    };

    out.innerHTML = `
      <div><strong>Rows:</strong> ${data.rows}</div>
      <div><strong>Columns:</strong> ${columns.join(", ")}</div>
      <div class="panel">
        <h3>Text Columns</h3>
        ${columns.map((c) => mkCheckbox(c, (data.default_text || []).includes(c), "text")).join("<br>")}
      </div>
      <div class="panel">
        <h3>Metadata Columns</h3>
        ${columns.map((c) => mkCheckbox(c, (data.default_meta || []).includes(c), "meta")).join("<br>")}
      </div>
      <button id="save-selection" class="btn">Save Selection</button>
      <div class="codebox">${safeJson({ dtypes: data.dtypes, preview: data.preview })}</div>
    `;

    el("save-selection").addEventListener("click", async () => {
      const textColumns = Array.from(out.querySelectorAll('input[data-kind="text"]:checked')).map((n) => n.value);
      const metaColumns = Array.from(out.querySelectorAll('input[data-kind="meta"]:checked')).map((n) => n.value);
      const resp = await postJson("/api/selection", {
        text_columns: textColumns,
        meta_columns: metaColumns,
      });
      out.insertAdjacentHTML("beforeend", `<div class="codebox">Saved selection: ${safeJson(resp)}</div>`);
    });
  });
}

async function setupIngest() {
  const btn = el("start-ingest");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    await fetch("/api/ingest/start", { method: "POST" });
    const log = el("event-log");
    log.textContent = "";

    const source = new EventSource("/api/ingest/stream");
    source.onmessage = (msg) => {
      const evt = JSON.parse(msg.data);
      renderStage("stage-list", evt);
      log.textContent += `${new Date(evt.ts * 1000).toLocaleTimeString()} | ${evt.stage} | ${evt.status}\n${safeJson(evt.detail)}\n\n`;
      log.scrollTop = log.scrollHeight;
      if (evt.stage === "Pipeline" && ["done", "error"].includes(evt.status)) {
        source.close();
      }
    };
  });
}

let chunkPage = 1;
async function loadChunks() {
  const holder = el("chunks-list");
  if (!holder) return;

  const q = encodeURIComponent(el("f-q")?.value || "");
  const priority = encodeURIComponent(el("f-priority")?.value || "");
  const moduleV = encodeURIComponent(el("f-module")?.value || "");
  const jira = encodeURIComponent(el("f-jira")?.value || "");

  const res = await fetch(`/api/chunks?page=${chunkPage}&q=${q}&priority=${priority}&module=${moduleV}&jira_id=${jira}`);
  const data = await res.json();

  el("page-label").textContent = `Page ${data.page} / ${Math.max(1, Math.ceil(data.total / data.page_size))}`;

  holder.innerHTML = (data.rows || []).map((r) => `
    <article class="chunk-card ${r.highlighted ? "highlight" : ""}">
      <div><strong>Chunk:</strong> ${r.chunk_id}</div>
      <div><strong>Payload:</strong> ${safeJson(r.payload)}</div>
      <div><strong>Dense preview:</strong> ${safeJson(r.dense_preview)}</div>
      <div><strong>Sparse preview:</strong> ${safeJson(r.sparse_preview)}</div>
      <div><strong>Text:</strong><br>${(r.text || "").replaceAll("<", "&lt;")}</div>
    </article>
  `).join("");
}

async function setupChunks() {
  if (!el("chunks-list")) return;

  el("apply-filter")?.addEventListener("click", () => {
    chunkPage = 1;
    loadChunks();
  });

  el("prev-page")?.addEventListener("click", () => {
    chunkPage = Math.max(1, chunkPage - 1);
    loadChunks();
  });

  el("next-page")?.addEventListener("click", () => {
    chunkPage += 1;
    loadChunks();
  });

  loadChunks();
}

function addChatMessage(role, text) {
  const box = el("chat-box");
  if (!box) return;
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.innerHTML = `<strong>${role === "user" ? "You" : "Assistant"}</strong><br>${formatMessage(text)}`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  if (role === "assistant") {
    lastAssistantText = text;
  }
}

async function setupTopbarHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (!data.ok) return;

    if (el("pill-cases")) el("pill-cases").textContent = `Cases: ${data.chunks_indexed ?? 0}`;
    if (el("pill-llm")) el("pill-llm").textContent = `LLM: ${data.llm_provider || "--"}`;
    if (el("pill-db")) el("pill-db").textContent = `DB: ${data.qdrant_mode || "--"}`;
    if (el("subtle-cases")) el("subtle-cases").textContent = `Indexed Cases: ${data.chunks_indexed ?? 0}`;
  } catch (_) {
    // No-op, keep UI usable even if health check fails.
  }
}

async function setupChat() {
  const form = el("chat-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = el("question");
    const question = input.value.trim();
    if (!question) return;

    addChatMessage("user", question);
    input.value = "";
    const sendBtn = form.querySelector("button[type='submit']");
    if (sendBtn) {
      sendBtn.disabled = true;
      sendBtn.textContent = "Thinking...";
    }

    const dbg = el("chat-debug");
    dbg.textContent = "Running pipeline...";

    const stages = ["Rewrite", "Dense + Sparse", "RRF", "Rerank", "Generate"];
    stages.forEach((s) => renderStage("chat-stage-list", { stage: s, status: "running" }));

    let res;
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 45000);
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });
      clearTimeout(timeout);
      res = await response.json();
      if (res.error) {
        addChatMessage("assistant", `Error: ${res.error}`);
        return;
      }
    } catch (err) {
      addChatMessage(
        "assistant",
        "Request timed out. Try a narrower query (module + priority) or retry in a few seconds."
      );
      return;
    } finally {
      if (sendBtn) {
        sendBtn.disabled = false;
        sendBtn.textContent = "Send";
      }
    }

    stages.forEach((s) => renderStage("chat-stage-list", { stage: s, status: "done" }));

    dbg.textContent = safeJson({
      rewrites: res.rewrites,
      dense_top: res.dense_top,
      sparse_top: res.sparse_top,
      rerank: res.rerank,
      citations: res.citations,
    });

    addChatMessage("assistant", res.answer);
  });

  el("export-jira")?.addEventListener("click", async () => {
    if (!lastAssistantText.trim()) {
      addChatMessage("assistant", "No assistant response available yet for export.");
      return;
    }

    const res = await fetch("/api/chat/export_jira", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: lastAssistantText, issue_key: "VWO-NEW" }),
    });

    if (!res.ok) {
      addChatMessage("assistant", "Export failed. Try again after generating a structured test case.");
      return;
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "generated_jira_row.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });

  el("copy-answer")?.addEventListener("click", async () => {
    if (!lastAssistantText.trim()) {
      showToast("No answer to copy yet");
      return;
    }
    try {
      await navigator.clipboard.writeText(lastAssistantText);
      showToast("Answer copied");
    } catch (_) {
      showToast("Copy failed");
    }
  });

  el("clear-chat")?.addEventListener("click", () => {
    const box = el("chat-box");
    if (box) box.innerHTML = "";
    lastAssistantText = "";
  });

  document.querySelectorAll(".qp").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-q") || "";
      const input = el("question");
      if (input) {
        input.value = q;
        input.focus();
      }
    });
  });
}

setupTopbarHealth();
setupUpload();
setupIngest();
setupChunks();
setupChat();
