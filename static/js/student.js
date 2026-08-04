// Student management page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Load students on page load
    loadStudents();

    // Event listeners
    document.getElementById('enroll-form').addEventListener('submit', handleEnroll);
    document.getElementById('image-upload').addEventListener('change', handleImageUpload);
    document.getElementById('additional-image-upload').addEventListener('change', handleAdditionalImageUpload);
    document.getElementById('upload-images-btn').addEventListener('click', uploadAdditionalImages);
    document.getElementById('class-filter').addEventListener('change', filterStudents);
    document.getElementById('search-input').addEventListener('input', filterStudents);

    // Modal buttons
    document.getElementById('mark-present-btn').addEventListener('click', () => markAttendance('present'));
    document.getElementById('mark-absent-btn').addEventListener('click', () => markAttendance('absent'));
    document.getElementById('delete-student-btn').addEventListener('click', deleteStudent);
    document.getElementById('refresh-chart-btn').addEventListener('click', refreshChart);

    // Setup drag & drop
    setupDragDrop('file-upload-area', 'image-upload', handleImageUpload);
    setupDragDrop('additional-file-upload-area', 'additional-image-upload', handleAdditionalImageUpload);

    // Table action delegation (view/edit)
    document.getElementById('students-table-body').addEventListener('click', function(e) {
        if (e.target.classList.contains('view-btn')) {
            const id = parseInt(e.target.dataset.studentId);
            viewStudent(id);
        }
        if (e.target.classList.contains('edit-btn')) {
            const id = parseInt(e.target.dataset.studentId);
            editStudent(id);
        }
    });
});

let students = [];
let selectedStudentId = null;
let attendanceChart = null;
let currentFiles = [];
let additionalFiles = [];

async function loadStudents() {
    try {
        const resp = await fetch('/api/students/all'); 
        const data = await resp.json();
        students = data.students || [];
        populateTable();
        populateStudentSelect();
    } catch (e) {
        console.error('Load students error:', e);
    }
}

function populateTable() {
    const tbody = document.getElementById('students-table-body');
    tbody.innerHTML = '';
    students.forEach(s => {
        const lastAtt = s.last_attendance ? new Date(s.last_attendance).toLocaleString() : 'Never';
        const statusClass = s.today_status === 'present' ? 'status-present' : 'status-absent';
        const statusText = s.today_status === 'present' ? 'Present' : 'Absent';
        const avatar = s.avatar_url || `https://ui-avatars.com/api/?name=${encodeURIComponent(s.name)}&background=4f46e5&color=fff`;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><img src="${avatar}" alt="${s.name}" style="width:32px;height:32px;border-radius:50%;margin-right:0.5rem;"> ${s.name}</td>
            <td>${s.id}</td>
            <td>${s.name}</td>
            <td>${s.department}</td>
            <td>${s.images_count} images</td>
            <td>${lastAtt}</td>
            <td><span class="status-badge ${statusClass}">${statusText}</span></td>
            <td>
                <button class="btn btn-sm btn-outline view-btn" data-student-id="${s.id}">View</button>
                <button class="btn btn-sm btn-secondary edit-btn" data-student-id="${s.id}">Edit</button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function populateStudentSelect() {
    const sel = document.getElementById('student-select');
    sel.innerHTML = '<option value="">Select student</option>';
    students.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = `${s.name} (${s.id})`;
        sel.appendChild(opt);
    });
}

function filterStudents() {
    const classFilter = document.getElementById('class-filter').value;
    const search = document.getElementById('search-input').value.toLowerCase();
    const filtered = students.filter(s => {
        const matchClass = !classFilter || s.department === classFilter;
        const matchSearch = !search || s.name.toLowerCase().includes(search) || s.id.toString().includes(search);
        return matchClass && matchSearch;
    });
    const tbody = document.getElementById('students-table-body');
    tbody.innerHTML = '';
    filtered.forEach(s => {
        // same as populateTable but filtered
        const lastAtt = s.last_attendance ? new Date(s.last_attendance).toLocaleString() : 'Never';
        const statusClass = s.today_status === 'present' ? 'status-present' : 'status-absent';
        const statusText = s.today_status === 'present' ? 'Present' : 'Absent';
        const avatar = s.avatar_url || `https://ui-avatars.com/api/?name=${encodeURIComponent(s.name)}&background=4f46e5&color=fff`;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><img src="${avatar}" alt="${s.name}" style="width:32px;height:32px;border-radius:50%;margin-right:0.5rem;"> ${s.name}</td>
            <td>${s.id}</td>
            <td>${s.department}</td>
            <td>${s.images_count} images</td>
            <td>${lastAtt}</td>
            <td><span class="status-badge ${statusClass}">${statusText}</span></td>
            <td>
                <button class="btn btn-sm btn-outline view-btn" data-student-id="${s.id}">View</button>
                <button class="btn btn-sm btn-secondary edit-btn" data-student-id="${s.id}">Edit</button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Image upload handlers
function handleImageUpload(e) {
    const files = e.target.files;
    const container = document.getElementById('image-preview-container');
    for (let f of files) {
        if (currentFiles.length >= 3) break;
        currentFiles.push(f);
        const reader = new FileReader();
        reader.onload = (ev) => {
            const div = document.createElement('div');
            div.style.position = 'relative';
            div.style.display = 'inline-block';
            const img = document.createElement('img');
            img.src = ev.target.result;
            img.className = 'image-preview';
            const remove = document.createElement('button');
            remove.className = 'remove-image';
            remove.innerHTML = '&times;';
            remove.onclick = () => {
                div.remove();
                const idx = currentFiles.indexOf(f);
                if (idx > -1) currentFiles.splice(idx, 1);
            };
            div.appendChild(img);
            div.appendChild(remove);
            container.appendChild(div);
        };
        reader.readAsDataURL(f);
    }
    e.target.value = '';
}

function handleAdditionalImageUpload(e) {
    const files = e.target.files;
    const container = document.getElementById('additional-image-preview-container');
    for (let f of files) {
        if (additionalFiles.length >= 3) break;
        additionalFiles.push(f);
        const reader = new FileReader();
        reader.onload = (ev) => {
            const div = document.createElement('div');
            div.style.position = 'relative';
            div.style.display = 'inline-block';
            const img = document.createElement('img');
            img.src = ev.target.result;
            img.className = 'image-preview';
            const remove = document.createElement('button');
            remove.className = 'remove-image';
            remove.innerHTML = '&times;';
            remove.onclick = () => {
                div.remove();
                const idx = additionalFiles.indexOf(f);
                if (idx > -1) additionalFiles.splice(idx, 1);
            };
            div.appendChild(img);
            div.appendChild(remove);
            container.appendChild(div);
        };
        reader.readAsDataURL(f);
    }
    e.target.value = '';
}

// Enroll form submit
async function handleEnroll(e) {
    e.preventDefault();
    if (currentFiles.length === 0) {
        alert('Please upload at least one image.');
        return;
    }
    const formData = new FormData();
    formData.append('name', document.getElementById('full-name').value);
    formData.append('person_id', document.getElementById('student-id').value);
    formData.append('department', document.getElementById('branch').value);
    formData.append('email', document.getElementById('email').value);
    currentFiles.forEach(f => formData.append('images', f));

    try {
        const resp = await fetch('/api/enroll/', {
            method: 'POST',
            body: formData
        });
        if (resp.ok) {
            alert('Student enrolled successfully!');
            document.getElementById('enroll-form').reset();
            document.getElementById('image-preview-container').innerHTML = '';
            currentFiles = [];
            loadStudents();
            showTab('all-students');
        } else {
            alert('Enrollment failed.');
        }
    } catch (err) {
        console.error(err);
        alert('Error enrolling student.');
    }
}

// Upload additional images
async function uploadAdditionalImages() {
    const studentId = document.getElementById('student-select').value;
    if (!studentId) { alert('Select a student.'); return; }
    if (additionalFiles.length === 0) { alert('Select images.'); return; }

    const formData = new FormData();
    formData.append('student_id', studentId);
    additionalFiles.forEach(f => formData.append('images', f));

    try {
        const resp = await fetch('/api/upload-images/', {
            method: 'POST',
            body: formData
        });
        if (resp.ok) {
            alert('Images uploaded.');
            document.getElementById('additional-image-preview-container').innerHTML = '';
            additionalFiles = [];
            loadStudents();
        } else {
            alert('Upload failed.');
        }
    } catch (err) {
        console.error(err);
        alert('Error uploading images.');
    }
}

// View student
async function viewStudent(id) {
    try {
        const resp = await fetch(`/api/students/${id}/`);
        const s = await resp.json();
        selectedStudentId = id;
        document.getElementById('modal-student-name').textContent = s.name;
        document.getElementById('modal-student-id').textContent = s.id;
        document.getElementById('modal-student-branch').textContent = s.department;
        document.getElementById('modal-student-email').textContent = s.email;
        document.getElementById('modal-enrollment-date').textContent = new Date(s.enrollment_date).toLocaleDateString();
        document.getElementById('modal-images-count').textContent = s.images_count;
        document.getElementById('modal-today-status').textContent = s.today_status.charAt(0).toUpperCase() + s.today_status.slice(1);
        createAttendanceChart(s.attendance_data);
        document.getElementById('student-detail-modal').style.display = 'flex';
    } catch (err) {
        console.error(err);
        alert('Error loading student details.');
    }
}

function createAttendanceChart(data) {
    const ctx = document.getElementById('attendance-chart').getContext('2d');
    if (attendanceChart) attendanceChart.destroy();
    const labels = data.labels || [];
    const values = data.data || [];
    attendanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Attendance',
                data: values,
                backgroundColor: values.map(v => v > 0 ? 'rgba(16,185,129,0.7)' : 'rgba(239,68,68,0.7)'),
                borderColor: values.map(v => v > 0 ? 'rgb(16,185,129)' : 'rgb(239,68,68)'),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } }
        }
    });
}

async function markAttendance(status) {
    if (!selectedStudentId) return;
    try {
        const resp = await fetch(`/api/students/${selectedStudentId}/attendance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status })
        });
        if (resp.ok) {
            alert(`Marked as ${status}`);
            closeModal();
            loadStudents();
        } else {
            alert('Failed to mark attendance.');
        }
    } catch (err) {
        console.error(err);
        alert('Error.');
    }
}

async function deleteStudent() {
    if (!selectedStudentId) return;
    if (!confirm('Delete this student?')) return;
    try {
        const resp = await fetch(`/api/students/${selectedStudentId}/delete`, { method: 'DELETE' });
        if (resp.ok) {
            alert('Deleted.');
            closeModal();
            loadStudents();
        } else {
            alert('Delete failed.');
        }
    } catch (err) {
        console.error(err);
        alert('Error.');
    }
}

async function refreshChart() {
    if (!selectedStudentId) return;
    try {
        const resp = await fetch(`/api/students/${selectedStudentId}/attendance`);
        const s = await resp.json();
        createAttendanceChart(s.attendance_data);
    } catch (err) {
        console.error(err);
    }
}

function editStudent(id) {
    alert(`Edit student ${id} (implement as needed)`);
}

function setupDragDrop(areaId, inputId, handler) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    ['dragenter','dragover'].forEach(ev => area.addEventListener(ev, e => { e.preventDefault(); area.classList.add('drag-over'); }));
    ['dragleave','drop'].forEach(ev => area.addEventListener(ev, e => { e.preventDefault(); area.classList.remove('drag-over'); }));
    area.addEventListener('drop', e => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        const event = new Event('change');
        Object.defineProperty(event, 'target', { value: { files } });
        handler(event);
    });
}