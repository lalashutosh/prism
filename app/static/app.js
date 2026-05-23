const statusEl = document.querySelector("#status");
const messagesEl = document.querySelector("#messages");
const chatForm = document.querySelector("#chatForm");
const questionInput = document.querySelector("#questionInput");

function setStatus(text) {
  statusEl.textContent = text;
}

function addMessage(role, text, details = {}) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.textContent = text;

  if (details.sources?.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    for (const source of details.sources) {
      const item = document.createElement("div");
      item.className = "source";
      const pageText = source.page_start === source.page_end
        ? `page ${source.page_start}`
        : `pages ${source.page_start}-${source.page_end}`;
      item.innerHTML = `<div class="source-title">${source.document_name}, ${pageText}</div><div>${escapeHtml(source.text)}</div>`;
      sources.appendChild(item);
    }
    node.appendChild(sources);
  }

  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  questionInput.value = "";
  addMessage("user", question);
  setStatus("Thinking...");

  const submitButton = chatForm.querySelector("button");
  submitButton.disabled = true;
  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: 6,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      addMessage("assistant", data.detail || "Query failed.");
      setStatus("Ready.");
      return;
    }
    addMessage("assistant", data.answer, { sources: data.sources });
    setStatus("Ready.");
  } catch (error) {
    addMessage("assistant", String(error));
    setStatus("Ready.");
  } finally {
    submitButton.disabled = false;
  }
});
