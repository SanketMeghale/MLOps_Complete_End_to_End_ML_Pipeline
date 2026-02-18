async function checkSpam() {
    const msg = document.getElementById("message").value;
    const result = document.getElementById("result");
    const status = document.getElementById("status");
    const confidence = document.getElementById("confidence");
    const fill = document.getElementById("fill");

    if (!msg.trim()) return;

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
    });

    const data = await res.json();

    result.classList.remove("hidden");
    status.innerText =
        data.prediction === "Spam" ? "🚨 Spam Detected" : "✅ Not Spam";
    confidence.innerText = `Confidence: ${data.confidence}%`;

    fill.style.width = data.confidence + "%";
    fill.style.background =
        data.prediction === "Spam" ? "red" : "lime";
}
