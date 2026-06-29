/**
 * SmartQ ML Clinical Playground - Application Logic
 * Integrates the front-end forms, presets, simulator states, and FastAPI back-end.
 */

// Preset Cases Database
const PRESETS = {
    asthma: {
        age: 25,
        sex: "M",
        language: "English",
        mental_status_triage: "alert",
        symptoms: "cannot breathe, severe wheezing and cough, chest tightness, blue lips, struggling for air",
        chief_complaint_system: "respiratory",
        pain_score: 4,
        temperature_c: 37.2,
        heart_rate: 115,
        respiratory_rate: 28,
        spo2: 88,
        systolic_bp: 120,
        diastolic_bp: 80,
        gcs_total: 15,
        news2_score: 5,
        num_prior_ed_visits_12m: 2,
        num_prior_admissions_12m: 1
    },
    chestPain: {
        age: 62,
        sex: "M",
        language: "English",
        mental_status_triage: "alert",
        symptoms: "sudden crushing chest pain radiating to left arm and jaw, sweating, palpitations, feel heavy pressure in chest like an elephant sitting on it",
        chief_complaint_system: "cardiac",
        pain_score: 9,
        temperature_c: 36.8,
        heart_rate: 105,
        respiratory_rate: 20,
        spo2: 94,
        systolic_bp: 142,
        diastolic_bp: 91,
        gcs_total: 15,
        news2_score: 3,
        num_prior_ed_visits_12m: 1,
        num_prior_admissions_12m: 0
    },
    stroke: {
        age: 71,
        sex: "F",
        language: "English",
        mental_status_triage: "drowsy",
        symptoms: "sudden onset of slurred speech, visible face droop on the right side, numbness and one sided weakness in right arm and right leg since this morning",
        chief_complaint_system: "neurological",
        pain_score: 0,
        temperature_c: 36.6,
        heart_rate: 82,
        respiratory_rate: 16,
        spo2: 96,
        systolic_bp: 168,
        diastolic_bp: 95,
        gcs_total: 12,
        news2_score: 3,
        num_prior_ed_visits_12m: 0,
        num_prior_admissions_12m: 0
    },
    pediatric: {
        age: 4,
        sex: "F",
        language: "Spanish",
        mental_status_triage: "alert",
        symptoms: "child has a fever for 2 days, runny nose, slight dry cough, minor body aches, eating normally, active but warm to touch",
        chief_complaint_system: "respiratory",
        pain_score: 3,
        temperature_c: 39.2,
        heart_rate: 122,
        respiratory_rate: 22,
        spo2: 98,
        systolic_bp: 95,
        diastolic_bp: 60,
        gcs_total: 15,
        news2_score: 2,
        num_prior_ed_visits_12m: 0,
        num_prior_admissions_12m: 0
    },
    sprain: {
        age: 19,
        sex: "M",
        language: "English",
        mental_status_triage: "alert",
        symptoms: "twisted right ankle playing soccer, swelling, bruised, painful to touch, unable to put full weight on it, no open wound",
        chief_complaint_system: "trauma",
        pain_score: 5,
        temperature_c: 36.5,
        heart_rate: 76,
        respiratory_rate: 14,
        spo2: 99,
        systolic_bp: 118,
        diastolic_bp: 75,
        gcs_total: 15,
        news2_score: 0,
        num_prior_ed_visits_12m: 1,
        num_prior_admissions_12m: 0
    }
};

// Default Simulation Clinics Queue State
const DEFAULT_QUEUES = [
    { route: "Cardiology", currentQueueLength: 3, availableDoctors: 1 },
    { route: "Pulmonology", currentQueueLength: 2, availableDoctors: 1 },
    { route: "Neurology", currentQueueLength: 1, availableDoctors: 1 },
    { route: "Gastroenterology", currentQueueLength: 4, availableDoctors: 1 },
    { route: "Orthopaedics", currentQueueLength: 5, availableDoctors: 2 },
    { route: "Paediatrics", currentQueueLength: 2, availableDoctors: 1 },
    { route: "General OPD", currentQueueLength: 8, availableDoctors: 3, acceptsFallback: true }
];

// In-Memory Queue State
let activeQueues = JSON.parse(JSON.stringify(DEFAULT_QUEUES));

// KTAS Description mapping
const KTAS_INFO = {
    1: { title: "Level 1 — Resuscitation", desc: "Immediate assessment. Life-threatening condition requiring active resuscitation.", color: "var(--ktas-1)" },
    2: { title: "Level 2 — Emergency", desc: "Assess within 15 minutes. High-risk condition requiring rapid intervention.", color: "var(--ktas-2)" },
    3: { title: "Level 3 — Urgent", desc: "Assess within 30 minutes. Stable but serious condition requiring workup.", color: "var(--ktas-3)" },
    4: { title: "Level 4 — Less Urgent", desc: "Assess within 60 minutes. Minor or chronic conditions, mild discomfort.", color: "var(--ktas-4)" },
    5: { title: "Level 5 — Non-Urgent", desc: "Assess within 120 minutes. Routine clinical concerns, standard timelines.", color: "var(--ktas-5)" }
};

document.addEventListener("DOMContentLoaded", () => {
    // Initialize icons
    lucide.createIcons();

    // Health check on backend
    checkAPIHealth();

    // Render Queue State Simulator UI
    renderQueueSimulator();

    // Event Listeners: Presets
    document.querySelectorAll(".preset-btn[data-preset]").forEach(btn => {
        btn.addEventListener("click", (e) => {
            const presetName = btn.getAttribute("data-preset");
            loadPreset(presetName);
            
            // Toggle active styling
            document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
        });
    });

    // Reset button
    document.getElementById("preset-reset").addEventListener("click", () => {
        document.getElementById("triage-form").reset();
        document.getElementById("pain_score-val").textContent = "5";
        document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        hideResults();
    });

    // Pain Score Slider Sync
    const painSlider = document.getElementById("pain_score");
    const painVal = document.getElementById("pain_score-val");
    painSlider.addEventListener("input", (e) => {
        painVal.textContent = e.target.value;
    });

    // NEWS2 Score Calculator Button
    document.getElementById("btn-calc-news2").addEventListener("click", () => {
        const score = calculateNEWS2();
        document.getElementById("news2_score").value = score;
    });

    // Reset Queues Button
    document.getElementById("btn-reset-queues").addEventListener("click", () => {
        activeQueues = JSON.parse(JSON.stringify(DEFAULT_QUEUES));
        renderQueueSimulator();
    });

    // Form Submit
    document.getElementById("triage-form").addEventListener("submit", (e) => {
        e.preventDefault();
        submitTriage();
    });

    // Tab Navigation
    document.querySelectorAll(".tab-btn").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));

            tab.classList.add("active");
            const targetId = `tab-${tab.getAttribute("data-tab")}`;
            document.getElementById(targetId).classList.add("active");
        });
    });

    // Copy Buttons
    document.getElementById("btn-copy-req").addEventListener("click", () => {
        copyTextToClipboard("json-request-display", "btn-copy-req");
    });
    document.getElementById("btn-copy-res").addEventListener("click", () => {
        copyTextToClipboard("json-response-display", "btn-copy-res");
    });
});

// Checks health of FastAPI endpoint
async function checkAPIHealth() {
    const badge = document.getElementById("api-status-badge");
    const indicator = document.getElementById("api-ping");
    const text = document.getElementById("api-status-text");

    try {
        const response = await fetch("/health");
        if (response.ok) {
            const data = await response.json();
            indicator.className = "status-indicator online";
            text.textContent = `ML Service Active (v${data.model_version})`;
            document.getElementById("app-model-version").textContent = data.model_version;
        } else {
            throw new Error("HTTP error");
        }
    } catch (err) {
        indicator.className = "status-indicator offline";
        text.textContent = "ML Service Offline";
        console.error("Health check failed:", err);
    }
}

// Load a preset case
function loadPreset(name) {
    const caseData = PRESETS[name];
    if (!caseData) return;

    // Fill form inputs
    document.getElementById("age").value = caseData.age;
    document.getElementById("sex").value = caseData.sex;
    document.getElementById("language").value = caseData.language;
    document.getElementById("mental_status_triage").value = caseData.mental_status_triage;
    document.getElementById("symptoms").value = caseData.symptoms;
    document.getElementById("chief_complaint_system").value = caseData.chief_complaint_system;
    document.getElementById("pain_score").value = caseData.pain_score;
    document.getElementById("pain_score-val").textContent = caseData.pain_score;
    document.getElementById("temperature_c").value = caseData.temperature_c;
    document.getElementById("heart_rate").value = caseData.heart_rate;
    document.getElementById("respiratory_rate").value = caseData.respiratory_rate;
    document.getElementById("spo2").value = caseData.spo2;
    document.getElementById("systolic_bp").value = caseData.systolic_bp;
    document.getElementById("diastolic_bp").value = caseData.diastolic_bp;
    document.getElementById("gcs_total").value = caseData.gcs_total;
    document.getElementById("news2_score").value = caseData.news2_score;
    document.getElementById("num_prior_ed_visits_12m").value = caseData.num_prior_ed_visits_12m;
    document.getElementById("num_prior_admissions_12m").value = caseData.num_prior_admissions_12m;
}

// Calculate clinical NEWS2 Score based on inputs
function calculateNEWS2() {
    let score = 0;

    // 1. Respiratory Rate
    const rr = parseInt(document.getElementById("respiratory_rate").value);
    if (!isNaN(rr)) {
        if (rr <= 8) score += 3;
        else if (rr >= 9 && rr <= 11) score += 1;
        else if (rr >= 12 && rr <= 20) score += 0;
        else if (rr >= 21 && rr <= 24) score += 2;
        else if (rr >= 25) score += 3;
    }

    // 2. SpO2
    const spo2 = parseInt(document.getElementById("spo2").value);
    if (!isNaN(spo2)) {
        if (spo2 <= 91) score += 3;
        else if (spo2 >= 92 && spo2 <= 93) score += 2;
        else if (spo2 >= 94 && spo2 <= 95) score += 1;
        else if (spo2 >= 96) score += 0;
    }

    // 3. Temperature
    const temp = parseFloat(document.getElementById("temperature_c").value);
    if (!isNaN(temp)) {
        if (temp <= 35.0) score += 3;
        else if (temp >= 35.1 && temp <= 36.0) score += 1;
        else if (temp >= 36.1 && temp <= 38.0) score += 0;
        else if (temp >= 38.1 && temp <= 39.0) score += 1;
        else if (temp >= 39.1) score += 2;
    }

    // 4. Heart Rate
    const hr = parseInt(document.getElementById("heart_rate").value);
    if (!isNaN(hr)) {
        if (hr <= 40) score += 3;
        else if (hr >= 41 && hr <= 50) score += 1;
        else if (hr >= 51 && hr <= 90) score += 0;
        else if (hr >= 91 && hr <= 110) score += 1;
        else if (hr >= 111 && hr <= 130) score += 2;
        else if (hr >= 131) score += 3;
    }

    // 5. Systolic BP
    const sbp = parseInt(document.getElementById("systolic_bp").value);
    if (!isNaN(sbp)) {
        if (sbp <= 90) score += 3;
        else if (sbp >= 91 && sbp <= 100) score += 2;
        else if (sbp >= 101 && sbp <= 110) score += 1;
        else if (sbp >= 111 && sbp <= 219) score += 0;
        else if (sbp >= 220) score += 3;
    }

    // 6. Consciousness (Mental Status)
    const mental = document.getElementById("mental_status_triage").value;
    const gcs = parseInt(document.getElementById("gcs_total").value);
    if (mental !== "alert" || (!isNaN(gcs) && gcs < 15)) {
        score += 3; // Altered consciousness or reduced GCS triggers 3 points
    }

    return score;
}

// Render the Queue Simulator Section in Sidebar
function renderQueueSimulator() {
    const container = document.getElementById("queue-simulator-container");
    container.innerHTML = "";

    activeQueues.forEach((q, index) => {
        // Average wait calculations: default is length * 8 mins / doctors
        const calculatedWait = Math.round((q.currentQueueLength * 8) / (q.availableDoctors || 0.5));
        
        const qDiv = document.createElement("div");
        qDiv.className = "sim-queue-item";
        qDiv.innerHTML = `
            <div class="sim-queue-header">
                <span class="sim-queue-name">${q.route}</span>
                ${q.route === "General OPD" ? "" : `<button type="button" class="sim-queue-remove" onclick="removeQueue(${index})" title="Remove clinic"><i data-lucide="trash-2" style="width: 14px; height: 14px;"></i></button>`}
            </div>
            <div class="sim-queue-controls">
                <div class="sim-control-group">
                    <span class="sim-val-label">
                        <span>Doctors:</span>
                        <strong id="sim-docs-val-${index}">${q.availableDoctors}</strong>
                    </span>
                    <input type="range" class="sim-input-slider" min="0" max="5" value="${q.availableDoctors}" oninput="updateQueueState(${index}, 'availableDoctors', this.value)">
                </div>
                <div class="sim-control-group">
                    <span class="sim-val-label">
                        <span>Queue:</span>
                        <strong id="sim-len-val-${index}">${q.currentQueueLength}</strong>
                    </span>
                    <input type="range" class="sim-input-slider" min="0" max="25" value="${q.currentQueueLength}" oninput="updateQueueState(${index}, 'currentQueueLength', this.value)">
                </div>
            </div>
        `;
        container.appendChild(qDiv);
    });
    lucide.createIcons();
}

// Update local simulator values
window.updateQueueState = function(index, key, val) {
    activeQueues[index][key] = parseInt(val);
    document.getElementById(`sim-${key === 'availableDoctors' ? 'docs' : 'len'}-val-${index}`).textContent = val;
};

// Remove a clinic from simulation
window.removeQueue = function(index) {
    activeQueues.splice(index, 1);
    renderQueueSimulator();
};

// Hides results pane and resets visual state
function hideResults() {
    document.getElementById("dashboard-empty-state").classList.remove("hidden");
    document.getElementById("dashboard-results").classList.add("hidden");
}

// Form Submission & API Integration
async function submitTriage() {
    const btn = document.getElementById("btn-submit-triage");
    btn.disabled = true;
    btn.innerHTML = `<span>Analyzing Vitals...</span> <i class="logo-icon animate-pulse" data-lucide="loader"></i>`;
    lucide.createIcons();

    // Build Request Payload
    const payload = {
        symptoms: document.getElementById("symptoms").value.trim()
    };

    // Helper to get number value or null
    const getNum = (id) => {
        const val = document.getElementById(id).value;
        return val === "" ? null : parseFloat(val);
    };

    // Helper to get string value or null
    const getStr = (id) => {
        const val = document.getElementById(id).value;
        return val === "" ? null : val;
    };

    // Patient Profile
    const age = getNum("age");
    if (age !== null) payload.age = Math.round(age);
    
    const sex = getStr("sex");
    if (sex) payload.sex = sex;
    
    payload.language = getStr("language");
    payload.mental_status_triage = getStr("mental_status_triage");
    
    // Complaint & Pain
    const complaint = getStr("chief_complaint_system");
    if (complaint) payload.chief_complaint_system = complaint;
    
    payload.pain_score = parseFloat(document.getElementById("pain_score").value);
    
    // Vitals
    payload.temperature_c = getNum("temperature_c");
    payload.heart_rate = getNum("heart_rate");
    payload.respiratory_rate = getNum("respiratory_rate");
    payload.spo2 = getNum("spo2");
    payload.systolic_bp = getNum("systolic_bp");
    payload.diastolic_bp = getNum("diastolic_bp");
    
    const gcs = getNum("gcs_total");
    if (gcs !== null) payload.gcs_total = Math.round(gcs);
    
    payload.news2_score = getNum("news2_score");
    
    // Load simulated queue states
    payload.availableRoutes = activeQueues.map(q => ({
        route: q.route,
        currentQueueLength: q.currentQueueLength,
        availableDoctors: q.availableDoctors,
        avgWaitMinutes: Math.round((q.currentQueueLength * 8) / (q.availableDoctors || 0.5)),
        acceptsFallback: q.acceptsFallback || false
    }));

    // Update Request JSON displays
    document.getElementById("json-request-display").textContent = JSON.stringify(payload, null, 2);

    try {
        const response = await fetch("/patient-flow", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errBody = await response.json();
            throw new Error(errBody.detail || "Server error running patient flow");
        }

        const data = await response.json();
        
        // Display Results
        document.getElementById("json-response-display").textContent = JSON.stringify(data, null, 2);
        renderVisualDashboard(data);
        
        // Switch tab to dashboard
        document.getElementById("tab-visual-btn").click();
        
        // Unhide
        document.getElementById("dashboard-empty-state").classList.add("hidden");
        document.getElementById("dashboard-results").classList.remove("hidden");
        
        // Scroll Results into view on mobile
        document.getElementById("dashboard-results").scrollIntoView({ behavior: "smooth" });

    } catch (err) {
        alert(`Error running triage: ${err.message}`);
        console.error("Triage analysis error:", err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span>Analyze Patient & Route</span> <i data-lucide="chevron-right"></i>`;
        lucide.createIcons();
    }
}

// Renders visual widgets in the Clinical Dashboard
function renderVisualDashboard(data) {
    // 1. KTAS Card priority classes
    const ktasCard = document.getElementById("ktas-card");
    const priority = data.priority.guardrailedPriorityClass;
    const confidenceVal = Math.round(data.priority.modelConfidence * 100);
    const modelPriority = data.priority.modelPriorityClass;
    const source = data.priority.source;

    // Reset KTAS priority styles
    ktasCard.className = `card ktas-result-card ktas-priority-${priority}`;
    
    // Set Level Circle
    const circle = document.getElementById("ktas-level-display");
    circle.className = `ktas-level-circle ktas-level-circle-${priority}`;
    circle.textContent = priority;

    // Set Text Labels
    const info = KTAS_INFO[priority];
    document.getElementById("ktas-priority-title").textContent = info.title;
    document.getElementById("ktas-priority-desc").textContent = `${info.desc} (${data.priority.guardrailedRecommendation})`;

    // Source Badge
    document.getElementById("badge-source").textContent = source;

    // Confidence Bar
    document.getElementById("confidence-percentage").textContent = `${confidenceVal}%`;
    const confBar = document.getElementById("confidence-bar");
    confBar.style.width = `${confidenceVal}%`;

    // Low confidence alert
    const lowConfAlert = document.getElementById("confidence-low-alert");
    if (data.priority.lowConfidence) {
        lowConfAlert.classList.remove("hidden");
    } else {
        lowConfAlert.classList.add("hidden");
    }

    // 2. Safety Overrides Card
    const safetyList = document.getElementById("safety-alerts-list");
    const safetyIcon = document.getElementById("safety-icon");
    safetyList.innerHTML = "";

    if (data.safety && data.safety.length > 0) {
        safetyIcon.className = "status-shield-icon alerted";
        safetyIcon.setAttribute("data-lucide", "shield-alert");
        
        data.safety.forEach(rule => {
            const ruleDiv = document.createElement("div");
            ruleDiv.className = `safety-alert-item ${rule.severity === 'critical' ? '' : 'warning-override'}`;
            ruleDiv.innerHTML = `
                <div class="alert-icon-container">
                    <i data-lucide="${rule.severity === 'critical' ? 'alert-octagon' : 'alert-triangle'}"></i>
                </div>
                <div class="alert-details">
                    <div class="alert-title-row">
                        <h4>${rule.ruleId.replace(/_/g, ' ')}</h4>
                        <span class="alert-severity-badge ${rule.severity}">${rule.severity}</span>
                    </div>
                    <p class="alert-desc">${rule.rationale} ${rule.forcedPriorityClass ? `<strong>(Override: KTAS Priority set to ${rule.forcedPriorityClass})</strong>` : ''}</p>
                </div>
            `;
            safetyList.appendChild(ruleDiv);
        });
    } else {
        safetyIcon.className = "status-shield-icon safe";
        safetyIcon.setAttribute("data-lucide", "shield-check");
        safetyList.innerHTML = `
            <div class="no-safety-triggers">
                <i data-lucide="check-circle" class="check-icon"></i>
                <p>No safety overrides or warning flags triggered. Patient flow runs on ML routing predictions.</p>
            </div>
        `;
    }

    // 3. Specialty Prediction Card
    document.getElementById("spec-primary").textContent = data.specialty.primarySpecialist;
    document.getElementById("spec-routed").textContent = data.specialty.routedSpecialty;
    document.getElementById("spec-reasoning").textContent = data.specialty.reasoning;

    // Alternatives
    const altsContainer = document.getElementById("spec-alternatives");
    altsContainer.innerHTML = "";
    if (data.specialty.alternatives && data.specialty.alternatives.length > 0) {
        data.specialty.alternatives.forEach(alt => {
            const confPercent = Math.round(alt.confidence * 100);
            const altDiv = document.createElement("div");
            altDiv.className = "alternative-item";
            altDiv.innerHTML = `
                <div class="alt-name-block">
                    <span class="alt-name">${alt.specialist}</span>
                    <span class="alt-route">queue: ${alt.routedSpecialty}</span>
                </div>
                <div class="alt-bar-section">
                    <div class="alt-bar-track">
                        <div class="alt-bar-fill" style="width: ${confPercent}%;"></div>
                    </div>
                    <span class="alt-val">${confPercent}%</span>
                </div>
            `;
            altsContainer.appendChild(altDiv);
        });
    } else {
        altsContainer.innerHTML = `<p style="font-size: 0.8rem; color: var(--text-muted); text-align: center;">No alternative fits calculated.</p>`;
    }

    // 4. Queue Assignment Card
    const qAssign = data.queueAssignment;
    document.getElementById("queue-assigned-name").textContent = qAssign.selectedRoute;
    document.getElementById("queue-rationale").textContent = qAssign.rationale;
    document.getElementById("queue-stat-length").textContent = `${qAssign.currentQueueLength} patients`;
    document.getElementById("queue-stat-doctors").textContent = qAssign.availableDoctors;

    const waitEl = document.getElementById("queue-stat-wait");
    if (qAssign.avgWaitMinutes !== null) {
        waitEl.textContent = `${Math.round(qAssign.avgWaitMinutes)}m`;
    } else {
        waitEl.textContent = "--";
    }

    // Route Badge styling
    const routeBadge = document.getElementById("queue-route-type");
    routeBadge.textContent = qAssign.routeType;
    if (qAssign.routeType === "safety_override") {
        routeBadge.className = "route-type-badge safety-override";
    } else if (qAssign.routeType === "fallback") {
        routeBadge.className = "route-type-badge fallback";
    } else {
        routeBadge.className = "route-type-badge";
    }

    // Target Queue banner accent matching KTAS priority for visual feedback
    const queueBanner = document.getElementById("queue-banner");
    queueBanner.style.borderColor = info.color;
    queueBanner.style.background = `rgba(${parseInt(info.color.substring(4,7)) || 99}, ${parseInt(info.color.substring(8,11)) || 102}, 241, 0.08)`;

    // 5. Diagnostic Tests Recommendations Card
    const testsGrid = document.getElementById("recommended-tests-grid");
    testsGrid.innerHTML = "";
    
    document.getElementById("tests-source").textContent = data.tests.source;

    if (data.tests.recommendations && data.tests.recommendations.length > 0) {
        data.tests.recommendations.forEach(rec => {
            const testDiv = document.createElement("div");
            testDiv.className = "test-card";
            testDiv.innerHTML = `
                <div class="test-header-row">
                    <span class="test-name">${rec.test}</span>
                    <span class="urgency-tag ${rec.urgency}">${rec.urgency}</span>
                </div>
                <p class="test-rationale">${rec.rationale}</p>
            `;
            testsGrid.appendChild(testDiv);
        });
    } else {
        testsGrid.innerHTML = `<p style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: var(--text-muted);">No lab test suggestions generated for this patient case.</p>`;
    }

    // Refresh icons
    lucide.createIcons();
}

// Copy Text to clipboard with button indicator feedback
function copyTextToClipboard(elementId, btnId) {
    const text = document.getElementById(elementId).textContent;
    const btn = document.getElementById(btnId);

    navigator.clipboard.writeText(text).then(() => {
        const originalText = btn.innerHTML;
        btn.innerHTML = `<i data-lucide="check"></i> Copied!`;
        btn.classList.add("copied");
        lucide.createIcons();

        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.remove("copied");
            lucide.createIcons();
        }, 2000);
    }).catch(err => {
        console.error("Failed to copy text:", err);
    });
}
