/**
 * Main JavaScript for Traffic Violation Detection System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Make table rows clickable
    const clickableRows = document.querySelectorAll('.table-hover tbody tr');
    if (clickableRows) {
        clickableRows.forEach(row => {
            row.addEventListener('click', function() {
                const detailsLink = this.querySelector('a.btn-outline-info');
                if (detailsLink) {
                    window.location.href = detailsLink.getAttribute('href');
                }
            });
        });
    }

    // Handle file input validation
    const videoInput = document.getElementById('video');
    if (videoInput) {
        videoInput.addEventListener('change', function() {
            const fileSize = this.files[0]?.size / 1024 / 1024; // size in MB
            const fileType = this.files[0]?.type;
            const submitButton = this.closest('form').querySelector('button[type="submit"]');
            
            // Check file size (max 50MB)
            if (fileSize > 50) {
                alert('File size exceeds the maximum allowed (50MB)');
                this.value = ''; // Clear the file input
                return;
            }
            
            // Check file type
            const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm'];
            if (this.files[0] && !validTypes.includes(fileType)) {
                alert('Invalid file type. Please upload MP4, AVI, MOV, or WEBM files only.');
                this.value = ''; // Clear the file input
                return;
            }
            
            // Enable submit button if file is valid
            if (submitButton && this.files[0]) {
                submitButton.disabled = false;
            }
        });
    }

    // Animation for violation cards on hover
    const violationCards = document.querySelectorAll('.violation-item');
    if (violationCards) {
        violationCards.forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.classList.add('shadow');
            });
            
            card.addEventListener('mouseleave', function() {
                this.classList.remove('shadow');
            });
        });
    }
});

/**
 * Format a timestamp in seconds to HH:MM:SS format
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Create a violation preview with thumbnails
 * @param {Array} violations - Array of violation objects
 * @param {string} containerId - ID of container element
 */
function createViolationPreview(violations, containerId) {
    if (!violations || violations.length === 0) return;
    
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear container
    container.innerHTML = '';
    
    // Create row
    const row = document.createElement('div');
    row.className = 'row g-2';
    
    // Add violation thumbnails
    violations.forEach(violation => {
        const col = document.createElement('div');
        col.className = 'col-md-3 col-sm-4 col-6';
        
        const card = document.createElement('div');
        card.className = 'card h-100 bg-dark border-secondary violation-thumb';
        card.addEventListener('click', () => {
            // You could implement a preview modal here
            console.log('Clicked violation:', violation);
        });
        
        let thumbnailHtml = '';
        if (violation.screenshot_path) {
            thumbnailHtml = `<img src="${violation.screenshot_path}" class="card-img-top" alt="Violation">`;
        } else {
            thumbnailHtml = `<div class="card-img-top bg-secondary d-flex align-items-center justify-content-center" style="height: 100px;">
                <i class="bi bi-camera-video-off text-light"></i>
            </div>`;
        }
        
        card.innerHTML = `
            ${thumbnailHtml}
            <div class="card-body p-2">
                <p class="card-text small mb-0">
                    ${formatTime(violation.timestamp)} - 
                    ${violation.violation_type === 'line_crossing' ? 'Line Crossing' : 
                      violation.violation_type === 'license_plate' ? 'License Issue' : 
                      violation.violation_type === 'not_yielding' ? 'Not Yielding' : 
                      violation.violation_type}
                </p>
            </div>
        `;
        
        col.appendChild(card);
        row.appendChild(col);
    });
    
    container.appendChild(row);
}
