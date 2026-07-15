const form = document.querySelector("#resumeForm");
const healthStatus = document.querySelector("#healthStatus");
const emptyState = document.querySelector("#emptyState");
const resultContent = document.querySelector("#resultContent");
const categoryResult = document.querySelector("#categoryResult");
const wordCount = document.querySelector("#wordCount");
const sourceName = document.querySelector("#sourceName");
const rankingList = document.querySelector("#rankingList");
const previewText = document.querySelector("#previewText");

async function checkHealth() {
    try {
        const response = await fetch("/api/health");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Model is not ready.");
        healthStatus.textContent = `Model ready: ${data.label_count} trained categories`;
        healthStatus.classList.add("ready");
    } catch (error) {
        healthStatus.textContent = error.message;
        healthStatus.classList.add("error");
    }
}

function setLoading(isLoading) {
    const button = form.querySelector("button");
    button.disabled = isLoading;
    button.textContent = isLoading ? "Analyzing..." : "Analyze Resume";
}

function renderRanking(items) {
    rankingList.innerHTML = "";
    if (!items || !items.length) {
        rankingList.innerHTML = "<p class=\"muted\">Ranking is unavailable for this classifier.</p>";
        return;
    }

    items.forEach((item) => {
        const row = document.createElement("div");
        row.className = "rank-row";
        row.innerHTML = `
            <div>
                <strong>${item.label}</strong>
                <span>Decision score ${item.score.toFixed(3)}</span>
            </div>
            <meter min="0" max="100" value="${item.confidence}"></meter>
            <b>${item.confidence.toFixed(1)}%</b>
        `;
        rankingList.appendChild(row);
    });
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setLoading(true);

    try {
        const formData = new FormData(form);
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed.");

        emptyState.classList.add("hidden");
        resultContent.classList.remove("hidden");
        categoryResult.textContent = data.category;
        wordCount.textContent = `${data.word_count} normalized words`;
        sourceName.textContent = `Source: ${data.source}`;
        previewText.textContent = data.preview || "No preview returned.";
        renderRanking(data.ranking);
    } catch (error) {
        emptyState.classList.remove("hidden");
        resultContent.classList.add("hidden");
        emptyState.innerHTML = `<span class="document-mark">!</span><p>${error.message}</p>`;
    } finally {
        setLoading(false);
    }
});

checkHealth();
