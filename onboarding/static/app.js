// Vitrine Onboarding wizard — vanilla JS, no framework.
// Drives a 6-step stepper and POSTs the collected manifest to /api/manifest.
// Secret containment: the pasted HF token value is sent once to the local
// server and is never shown in the review pane or in any response handling.

"use strict";

(function () {
  const TOTAL_STEPS = 6;
  let current = 1;

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const sections = $$(".step");
  const indicators = $$("#step-indicator li");
  const prevBtn = $("#prev");
  const nextBtn = $("#next");

  function showStep(n) {
    current = Math.max(1, Math.min(TOTAL_STEPS, n));
    sections.forEach((s) => {
      s.hidden = Number(s.dataset.step) !== current;
    });
    indicators.forEach((li) => {
      li.classList.toggle("active", Number(li.dataset.step) === current);
      li.classList.toggle("done", Number(li.dataset.step) < current);
    });
    prevBtn.hidden = current === 1;
    nextBtn.hidden = current === TOTAL_STEPS;
    if (current === TOTAL_STEPS) {
      renderReview();
    }
  }

  // ---- Objects: dynamic add/remove rows -----------------------------------

  let objectSeq = 0;
  const objectsList = $("#objects-list");

  function nextObjectId() {
    objectSeq += 1;
    return "obj-" + String(objectSeq).padStart(3, "0");
  }

  function addObjectRow(prefill) {
    const data = prefill || {};
    const row = document.createElement("fieldset");
    row.className = "object-row";
    const oid = data.id || nextObjectId();
    row.innerHTML = `
      <legend>Object</legend>
      <div class="field">
        <label>ID
          <input type="text" class="o-id" value="${escapeAttr(oid)}" />
        </label>
      </div>
      <div class="field">
        <label>Name
          <input type="text" class="o-name" value="${escapeAttr(data.name || "")}" placeholder="Reclining Figure" />
        </label>
      </div>
      <div class="field">
        <label>SAM3 concept
          <input type="text" class="o-concept" value="${escapeAttr(data.sam3_concept || "")}"
            placeholder="large bronze reclining human figure" />
        </label>
      </div>
      <div class="field">
        <label>Description
          <input type="text" class="o-desc" value="${escapeAttr(data.description || "")}"
            placeholder="Patinated bronze, ~2m, central plinth." />
        </label>
      </div>
      <div class="field two-col">
        <label>Priority
          <select class="o-priority">
            <option value="standard">standard</option>
            <option value="key">key</option>
          </select>
        </label>
        <label>Expected count
          <input type="number" class="o-count" min="1" step="1" value="${Number(data.expected_count) || 1}" />
        </label>
      </div>
      <button type="button" class="ghost danger remove-object">Remove</button>
    `;
    objectsList.appendChild(row);
    if (data.priority === "key") {
      row.querySelector(".o-priority").value = "key";
    }
    row.querySelector(".remove-object").addEventListener("click", () => {
      row.remove();
    });
  }

  function collectObjects() {
    return $$(".object-row").map((row) => ({
      id: (row.querySelector(".o-id").value || "").trim(),
      name: (row.querySelector(".o-name").value || "").trim(),
      sam3_concept: (row.querySelector(".o-concept").value || "").trim(),
      description: (row.querySelector(".o-desc").value || "").trim(),
      priority: row.querySelector(".o-priority").value,
      expected_count: Math.max(1, parseInt(row.querySelector(".o-count").value, 10) || 1),
    }));
  }

  // ---- Manifest assembly --------------------------------------------------

  function buildPayload() {
    const hfValueEl = $("#se-hf-value");
    const hfValue = hfValueEl.value.trim();

    const payload = {
      exhibit: {
        id: $("#ex-id").value.trim(),
        name: $("#ex-name").value.trim(),
        venue: $("#ex-venue").value.trim(),
        date: $("#ex-date").value.trim(),
        curator: $("#ex-curator").value.trim(),
        description: $("#ex-description").value.trim(),
      },
      drive: {
        url: $("#dr-url").value.trim(),
        rclone_remote: $("#dr-remote").value.trim() || "gdrive",
        recursive: $("#dr-recursive").checked,
      },
      objects: collectObjects(),
      secrets: {
        hf_token_env: $("#se-hf-env").value.trim() || "HF_TOKEN",
        gcloud_credentials_env: $("#se-gc-env").value.trim() || "GOOGLE_APPLICATION_CREDENTIALS",
        gcloud_project: $("#se-gc-project").value.trim(),
      },
      pipeline: {
        mesh_backend: $("#pi-mesh").value,
        matcher: $("#pi-matcher").value,
      },
      oversight: {
        backend: $("#ov-backend").value,
        artifact_vlm: $("#ov-vlm").value,
      },
    };

    if (hfValue) {
      payload.secrets.hf_token_value = hfValue;
    }
    return payload;
  }

  // Review pane never renders the raw token value.
  function renderReview() {
    const p = buildPayload();
    const safe = JSON.parse(JSON.stringify(p));
    if (safe.secrets && "hf_token_value" in safe.secrets) {
      safe.secrets.hf_token_value = "(set — contained server-side)";
    }
    safe.secrets.hf_token = "env:" + safe.secrets.hf_token_env;
    safe.secrets.gcloud_credentials = "env:" + safe.secrets.gcloud_credentials_env;
    $("#review").textContent = JSON.stringify(safe, null, 2);
  }

  // ---- Submit -------------------------------------------------------------

  async function generate() {
    const result = $("#result");
    const btn = $("#generate");
    btn.disabled = true;
    result.hidden = false;
    result.className = "result pending";
    result.textContent = "Generating…";

    try {
      const res = await fetch("/api/manifest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload()),
      });
      const body = await res.json();
      if (res.ok && body.ok) {
        result.className = "result success";
        const contained = body.secret_contained
          ? " A pasted token was contained server-side in .secrets.env (0600)."
          : "";
        result.textContent = "Manifest written to: " + body.toml_path + contained;
        // Clear the in-memory token field once accepted.
        $("#se-hf-value").value = "";
      } else {
        result.className = "result error";
        result.textContent = "Error: " + (body.error || res.statusText);
      }
    } catch (err) {
      result.className = "result error";
      result.textContent = "Request failed: " + err.message;
    } finally {
      btn.disabled = false;
    }
  }

  function validateStep(n) {
    if (n === 1) {
      for (const id of ["#ex-id", "#ex-name"]) {
        const el = $(id);
        if (!el.value.trim()) {
          el.focus();
          el.classList.add("invalid");
          return false;
        }
        el.classList.remove("invalid");
      }
    }
    return true;
  }

  // ---- Helpers ------------------------------------------------------------

  function escapeAttr(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // ---- Wiring -------------------------------------------------------------

  nextBtn.addEventListener("click", () => {
    if (validateStep(current)) {
      showStep(current + 1);
    }
  });
  prevBtn.addEventListener("click", () => showStep(current - 1));
  $("#add-object").addEventListener("click", () => addObjectRow());
  $("#generate").addEventListener("click", generate);

  indicators.forEach((li) => {
    li.addEventListener("click", () => {
      const target = Number(li.dataset.step);
      if (target <= current || validateStep(current)) {
        showStep(target);
      }
    });
  });

  // Seed one object row and show the first step.
  addObjectRow();
  showStep(1);
})();
