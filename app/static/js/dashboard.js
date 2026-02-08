// Dashboard SSE listener for real-time pipeline updates
document.addEventListener('DOMContentLoaded', function() {
    const statusDiv = document.getElementById('pipeline-status');
    const msgSpan = document.getElementById('pipeline-msg');
    if (!statusDiv || !msgSpan) return;

    // Connect to SSE for live updates
    const es = new EventSource('/api/stream');

    es.addEventListener('pipeline', function(e) {
        const data = JSON.parse(e.data);
        statusDiv.classList.remove('d-none');

        if (data.status === 'running') {
            statusDiv.className = 'alert alert-info mb-3';
            msgSpan.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>' +
                data.type + ' pipeline running...';
        } else if (data.status === 'completed') {
            statusDiv.className = 'alert alert-success mb-3';
            msgSpan.textContent = data.type + ' pipeline completed';
            setTimeout(() => location.reload(), 3000);
        } else if (data.status === 'failed') {
            statusDiv.className = 'alert alert-danger mb-3';
            msgSpan.textContent = data.type + ' pipeline failed: ' + (data.error || 'Unknown');
        }
    });

    es.onerror = function() {
        // Silently reconnect
        setTimeout(() => {
            es.close();
        }, 5000);
    };
});
