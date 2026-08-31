(() => {
  "use strict";

  const form = document.querySelector("#application-form");
  const input = document.querySelector("#application-number");
  const button = document.querySelector("#recognize-button");
  const loading = document.querySelector("#application-loading");
  const errorPanel = document.querySelector("#application-error");
  const resultSection = document.querySelector("#application-result");
  let detectedApiBase = "";

  function apiBase() {
    const fromQuery = new URLSearchParams(window.location.search).get("api");
    if (fromQuery) return fromQuery.replace(/\/$/, "");
    return detectedApiBase || window.location.origin;
  }

  async function detectApiBase() {
    if (new URLSearchParams(window.location.search).get("api")) return apiBase();
    if (window.location.protocol !== "http:" || window.location.port !== "5173") {
      detectedApiBase = window.location.origin;
      return detectedApiBase;
    }
    try {
      const response = await fetch(
        `${window.location.origin}/api/v1/mobile/config?probe=${Date.now()}`,
        { cache: "no-store" },
      );
      if (response.status !== 404 && response.status !== 501) {
        detectedApiBase = window.location.origin;
        return detectedApiBase;
      }
    } catch (_error) {
      // Static-only development mode falls back to the API port below.
    }
    const hostname = window.location.hostname.includes(":")
      ? `[${window.location.hostname}]`
      : window.location.hostname;
    detectedApiBase = `http://${hostname}:8000`;
    return detectedApiBase;
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderDocumentCard(document, index) {
    const card = element("article", "result-card");
    const header = element("header");
    header.append(
      element("strong", "", document.person_id || document.vehicle_id || `材料 ${index + 1}`),
      element("span", "", document.instance_id),
    );
    card.append(header);

    const fields = element("dl", "field-list");
    const entries = Object.entries(document.fields || {})
      .filter(([, value]) => String(value || "").trim());
    if (!entries.length) {
      fields.append(element("div", "empty-result", "未抽取到有效字段"));
    } else {
      entries.forEach(([name, value]) => {
        const row = element("div", "field-row");
        row.append(element("dt", "", name), element("dd", "", value));
        fields.append(row);
      });
    }
    card.append(fields);
    card.append(element("p", "source-files", `来源：${(document.source_files || []).join("、")}`));
    (document.warnings || []).forEach((warning) => {
      card.append(element("p", "card-warning", warning));
    });
    if (["unpaired_front", "unpaired_back"].includes(document.pairing_confidence)) {
      card.append(element("p", "card-warning", "身份证正反面未可靠配对，请人工确认"));
    }
    return card;
  }

  function renderColumn(key, documents) {
    const container = document.querySelector(`[data-document-results="${key}"]`);
    const count = document.querySelector(`[data-document-count="${key}"]`);
    container.replaceChildren();
    count.textContent = String(documents.length);
    if (!documents.length) {
      container.append(element("p", "empty-result", "该申请单未识别到此证件"));
      return;
    }
    documents.forEach((document, index) => {
      container.append(renderDocumentCard(document, index));
    });
  }

  function addChip(container, text, warning = false) {
    container.append(element("span", `summary-chip${warning ? " warning" : ""}`, text));
  }

  function renderResult(data) {
    document.querySelector("#result-application-number").textContent = `申请单 ${data.application_no}`;
    const summary = document.querySelector("#result-summary");
    summary.replaceChildren();
    addChip(summary, `${data.summary.person_count} 人`);
    addChip(
      summary,
      `${data.summary.id_card_count + data.summary.driver_license_count + data.summary.vehicle_license_count} 份证件`,
    );
    addChip(summary, `${data.summary.elapsed_seconds.toFixed(1)} 秒`);
    if (data.summary.missing_documents.length) {
      addChip(summary, `缺少：${data.summary.missing_documents.join("、")}`, true);
    }
    if (data.summary.error_count) addChip(summary, `${data.summary.error_count} 项失败`, true);

    renderColumn("id_cards", data.documents.id_cards || []);
    renderColumn("driver_licenses", data.documents.driver_licenses || []);
    renderColumn("vehicle_licenses", data.documents.vehicle_licenses || []);
    const galleryUrl = new URL("./application-files.html", window.location.href);
    galleryUrl.search = "";
    galleryUrl.searchParams.set("application_no", data.application_no);
    const configuredApi = new URLSearchParams(window.location.search).get("api");
    if (configuredApi) galleryUrl.searchParams.set("api", configuredApi);
    document.querySelector("#source-gallery-link").href = galleryUrl.toString();

    const validationSection = document.querySelector("#validation-section");
    const validationResults = document.querySelector("#validation-results");
    validationResults.replaceChildren();
    validationSection.hidden = !data.validations.length;
    data.validations.forEach((validation) => {
      const item = element(
        "div",
        `validation-item ${validation.severity === "warning" ? "warning" : ""}`,
      );
      item.append(
        element("span", "validation-dot"),
        element("span", "", validation.message),
      );
      validationResults.append(item);
    });

    const errorsSection = document.querySelector("#recognition-errors");
    const errors = document.querySelector("#recognition-error-list");
    errors.replaceChildren();
    errorsSection.hidden = !data.errors.length;
    data.errors.forEach((item) => {
      errors.append(element(
        "p",
        "recognition-error",
        `${item.document_type} · ${item.source_files.join("、")}：${item.error}`,
      ));
    });
    resultSection.hidden = false;
  }

  function errorMessage(detail) {
    if (typeof detail === "string") return detail;
    if (detail && typeof detail.message === "string") return detail.message;
    return "申请单识别失败，请检查服务日志";
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const applicationNo = input.value.trim();
    if (!/^\d{1,32}$/.test(applicationNo)) {
      errorPanel.textContent = "申请单号只能输入数字";
      errorPanel.hidden = false;
      return;
    }
    button.disabled = true;
    loading.hidden = false;
    errorPanel.hidden = true;
    resultSection.hidden = true;
    try {
      await detectApiBase();
      const response = await fetch(
        `${apiBase()}/api/v1/applications/${encodeURIComponent(applicationNo)}/recognition`,
        { method: "POST" },
      );
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorMessage(payload.detail));
      renderResult(payload);
    } catch (error) {
      errorPanel.textContent = error instanceof Error ? error.message : "申请单识别失败";
      errorPanel.hidden = false;
    } finally {
      button.disabled = false;
      loading.hidden = true;
    }
  });

  const initialApplicationNo = new URLSearchParams(window.location.search)
    .get("application_no");
  if (initialApplicationNo && /^\d{1,32}$/.test(initialApplicationNo)) {
    input.value = initialApplicationNo;
    form.requestSubmit();
  }
})();
