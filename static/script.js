function predict() {
    const input = document.getElementById("imageInput");
    const resultDiv = document.getElementById("result");
    const preview = document.getElementById("previewImage");

    if (input.files.length === 0) {
        alert("Please select an image first");
        return;
    }

    const file = input.files[0];

    // Show preview
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    const formData = new FormData();
    formData.append("file", file);

    resultDiv.innerHTML = "Predicting...";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = "Error: " + data.error;
        } else {
            resultDiv.innerHTML =
                `Prediction: ${data.prediction} <br> Confidence: ${data.confidence}%`;
        }
    })
    .catch(error => {
        resultDiv.innerHTML = "Something went wrong";
        console.error(error);
    });
}
