const form = document.querySelector("#paperForm");
const paperList = document.querySelector("#paperList");
const subjectList = document.querySelector("#subjectList");
const engineLabel = document.querySelector("#engineLabel");
const healthLine = document.querySelector("#healthLine");

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderSubjects(subjects) {
    subjectList.innerHTML = "";
    if (!subjects || !subjects.length) {
        subjectList.innerHTML = "<span class=\"empty-pill\">No subjects yet</span>";
        return;
    }
    subjectList.innerHTML = subjects.map((item) => `
        <span class="subject-pill">
            <b>${escapeHtml(item.label)}</b>
            ${item.confidence.toFixed(1)}%
        </span>
    `).join("");
}

function renderPaper(paper, index) {
    const terms = paper.terms.map((term) => `<span>${escapeHtml(term)}</span>`).join("");
    return `
        <article class="paper-card">
            <div class="score-block">
                <strong>${String(index + 1).padStart(2, "0")}</strong>
                <span>${paper.match_percent.toFixed(1)}%</span>
            </div>
            <div>
                <h3>${escapeHtml(paper.title)}</h3>
                <div class="terms">${terms}</div>
                <p>${escapeHtml(paper.abstract)}</p>
            </div>
        </article>
    `;
}

async function checkHealth() {
    try {
        const response = await fetch("/api/health");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Research model unavailable.");
        const mode = data.semantic_model ? "semantic recommender ready" : "TF-IDF fallback ready";
        healthLine.textContent = `${data.papers} papers indexed; ${data.embedding_dimensions}D embeddings; ${mode}`;
        healthLine.classList.add("ready");
    } catch (error) {
        healthLine.textContent = error.message;
        healthLine.classList.add("error");
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button");
    button.disabled = true;
    button.textContent = "Searching...";
    paperList.innerHTML = "<article class=\"empty-card\"><h3>Searching...</h3><p>Ranking papers and aggregating subject areas.</p></article>";

    try {
        const payload = {
            title: document.querySelector("#title").value,
            abstract: document.querySelector("#abstract").value,
            top_k: document.querySelector("#topK").value,
        };
        const response = await fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Recommendation failed.");

        engineLabel.textContent = data.engine.replace("-", " ");
        renderSubjects(data.subjects);
        paperList.innerHTML = data.recommendations.map(renderPaper).join("");
        if (data.runtime_note) {
            healthLine.textContent = data.runtime_note;
            healthLine.classList.add("warning");
        }
    } catch (error) {
        paperList.innerHTML = `<article class="empty-card error"><h3>Search failed</h3><p>${escapeHtml(error.message)}</p></article>`;
    } finally {
        button.disabled = false;
        button.textContent = "Recommend";
    }
});

renderSubjects([]);
checkHealth();
