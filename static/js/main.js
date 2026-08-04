// Common JavaScript functions

document.addEventListener('DOMContentLoaded', function() {
    // Tab switching (used in students page)
    window.showTab = function(tabId) {
        const tabs = document.querySelectorAll('.tab-content');
        tabs.forEach(t => t.style.display = 'none');
        document.getElementById(tabId).style.display = 'block';

        const tabLinks = document.querySelectorAll('.tab');
        tabLinks.forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(t => {
            if (t.textContent.toLowerCase().includes(tabId.replace('-', ' '))) {
                t.classList.add('active');
            }
        });
    };

    // Close modal (used in students)
    window.closeModal = function() {
        const modal = document.getElementById('student-detail-modal');
        if (modal) modal.style.display = 'none';
    };

    // Click outside modal to close
    document.addEventListener('click', function(e) {
        const modal = document.getElementById('student-detail-modal');
        if (modal && e.target === modal) {
            modal.style.display = 'none';
        }
    });
});