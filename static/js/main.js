const labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
];

function displayResults(data) {
    // Add debug logging
    console.log('Received data:', data);
    console.log('Mode:', data.mode);
    console.log('Has segmentation:', !!data.segmentation);
    
    const resultsDiv = document.getElementById('results');
    const predictionsDiv = document.getElementById('predictions');
    const jsonResponse = document.getElementById('jsonResponse');
    
    // Clear previous results
    resultsDiv.style.display = 'block';
    predictionsDiv.innerHTML = '';
    
    // Display classification results
    if (data.classification) {
        data.classification.forEach((prob, index) => {
            if (prob > 0.5) {
                const predDiv = document.createElement('div');
                predDiv.className = 'prediction';
                predDiv.innerHTML = `
                    <strong>${labels[index]}</strong>
                    <div class="probability">${(prob * 100).toFixed(2)}%</div>
                `;
                predictionsDiv.appendChild(predDiv);
            }
        });
    }
    
    // Display segmentation overlay if in segmentation mode
    if (data.mode === 'segmentation' && data.segmentation) {
        const segDiv = document.createElement('div');
        segDiv.className = 'segmentation';
        segDiv.innerHTML = `
            <h3>Segmentation Overlay</h3>
            <div class="segmentation-container">
                <img src="data:image/png;base64,${data.segmentation}" 
                     alt="Segmentation Overlay" 
                     style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px;">
            </div>
        `;
        resultsDiv.appendChild(segDiv);
    }
    
    // Display raw JSON response
    jsonResponse.textContent = JSON.stringify(data, null, 2);
}

document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const viewResults = document.getElementById('viewResults').checked;
    const mode = document.querySelector('input[name="mode"]:checked').value;
    formData.append('mode', mode);
    
    // Add debug logging
    console.log('Selected mode:', mode);
    console.log('FormData contents:', Object.fromEntries(formData));

    // Reset display
    document.getElementById('error').style.display = 'none';
    document.getElementById('results').style.display = 'none';

    if (viewResults) {
        // Use traditional form submission to get a page redirect
        const form = document.getElementById('uploadForm');
        form.method = 'POST';
        form.action = '/';
        form.submit();
    } else {
        // Use fetch and show results on same page
        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Error analyzing image');
                });
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = error.message;
            errorDiv.style.display = 'block';
        });
    }
}); 