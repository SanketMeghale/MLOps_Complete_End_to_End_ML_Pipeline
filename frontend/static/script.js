async function checkSpam() {
    const msg = document.getElementById("message").value.trim();
    const result = document.getElementById("result");
    const status = document.getElementById("status");
    const confidence = document.getElementById("confidence");
    const fill = document.getElementById("fill");

    // Safety check
    if (!msg) {
        alert("Please enter a message to analyze.");
        return;
    }

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: msg })
        });

        if (!res.ok) {
            throw new Error("Server error");
        }

        const data = await res.json();

        // Show result section
        result.classList.remove("hidden");

        // Status text
        if (data.prediction === "Spam") {
            status.innerText = "🚨 Spam Detected";
            status.style.color = "#ef4444"; // red
            fill.style.background = "#ef4444";
        } else {
            status.innerText = "✅ Not Spam";
            status.style.color = "#22c55e"; // green
            fill.style.background = "#22c55e";
        }

        // Confidence
        confidence.innerText = `Confidence: ${data.confidence}%`;

        // Progress bar animation
        fill.style.width = data.confidence + "%";

    } catch (error) {
        console.error(error);
        alert("Something went wrong. Please try again.");
    }
    result.classList.remove("hidden");
result.style.animation = "none";
result.offsetHeight; // trigger reflow
result.style.animation = "fadeUp 0.5s ease-out";
}