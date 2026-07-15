const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const healthStatus = document.querySelector("#healthStatus");
const engineNote = document.querySelector("#engineNote");
const promptButtons = document.querySelectorAll("[data-message]");

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function addMessage(role, text, detail = "") {
    const item = document.createElement("article");
    item.className = `message ${role}`;
    item.innerHTML = `
        <span>${role === "user" ? "You" : "Assistant"}</span>
        <p>${escapeHtml(text)}</p>
        ${detail ? `<small>${escapeHtml(detail)}</small>` : ""}
    `;
    messages.appendChild(item);
    messages.scrollTop = messages.scrollHeight;
}

async function checkHealth() {
    try {
        const response = await fetch("/api/health");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Chatbot unavailable.");
        const mode = data.transformer_model ? "fine-tuned T5 ready" : "retrieval fallback ready";
        healthStatus.textContent = `${mode}; ${data.retrieval_rows} support examples indexed`;
        healthStatus.classList.add("ready");
        engineNote.textContent = data.runtime_note || "";
    } catch (error) {
        healthStatus.textContent = error.message;
        healthStatus.classList.add("error");
    }
}

async function sendMessage(message) {
    addMessage("user", message);
    input.value = "";
    const button = form.querySelector("button");
    button.disabled = true;
    button.textContent = "Sending...";

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "The chatbot could not answer.");

        const detail = data.matched_query ? `Matched: ${data.matched_query}` : data.engine;
        addMessage("bot", data.response, detail);
        engineNote.textContent = data.runtime_note || `Engine: ${data.engine}`;
    } catch (error) {
        addMessage("bot", error.message);
    } finally {
        button.disabled = false;
        button.textContent = "Send";
        input.focus();
    }
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (message) sendMessage(message);
});

promptButtons.forEach((button) => {
    button.addEventListener("click", () => {
        input.value = button.dataset.message;
        sendMessage(button.dataset.message);
    });
});

checkHealth();
