const form = document.querySelector("#recommendForm");
const grid = document.querySelector("#productGrid");
const resultCount = document.querySelector("#resultCount");
const engineName = document.querySelector("#engineName");
const runtimeNote = document.querySelector("#runtimeNote");
const healthLine = document.querySelector("#healthLine");

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function money(value) {
    const text = String(value || "").trim();
    if (!text) return "Price unavailable";
    return /^\d+$/.test(text) ? `Rs. ${Number(text).toLocaleString("en-IN")}` : text;
}

function productCard(product) {
    const image = escapeHtml(product.image_url || "https://placehold.co/320x320/f4f6f4/1d2522?text=Mobile");
    const name = escapeHtml(product.name);
    return `
        <article class="product-card">
            <div class="image-wrap">
                <img src="${image}" alt="${name}" loading="lazy">
                <span>${product.match_percent.toFixed(1)}% match</span>
            </div>
            <div class="product-copy">
                <h2>${name}</h2>
                <div class="meta">
                    <b>${escapeHtml(money(product.price))}</b>
                    <span>${product.ratings.toFixed(1)} rating</span>
                </div>
                <p>${escapeHtml(product.specs)}</p>
            </div>
        </article>
    `;
}

async function checkHealth() {
    try {
        const response = await fetch("/api/health");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Catalog unavailable.");
        const mode = data.semantic_model ? "semantic model ready" : "TF-IDF fallback ready";
        healthLine.textContent = `${data.products} products loaded; ${mode}`;
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
    grid.innerHTML = "<article class=\"empty-card\"><h2>Searching...</h2><p>Ranking products against your query.</p></article>";

    try {
        const payload = {
            query: document.querySelector("#query").value,
            top_k: document.querySelector("#topK").value,
        };
        const response = await fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Recommendation failed.");

        resultCount.textContent = data.count;
        engineName.textContent = data.engine.replace("-", " ");
        runtimeNote.textContent = data.runtime_note || "";
        grid.innerHTML = data.recommendations.map(productCard).join("");
    } catch (error) {
        grid.innerHTML = `<article class="empty-card error"><h2>Search failed</h2><p>${escapeHtml(error.message)}</p></article>`;
    } finally {
        button.disabled = false;
        button.textContent = "Search";
    }
});

checkHealth();
