(() => {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const applicationNo = params.get("application_no") || "";
  const groupsContainer = document.querySelector("#source-groups");
  const loading = document.querySelector("#gallery-loading");
  const errorPanel = document.querySelector("#gallery-error");
  const viewer = document.querySelector("#image-viewer");
  const viewerImage = document.querySelector("#viewer-image");
  const viewerStage = document.querySelector("#viewer-stage");
  const viewerTitle = document.querySelector("#viewer-title");
  const viewerPosition = document.querySelector("#viewer-position");
  let detectedApiBase = "";
  let images = [];
  let currentIndex = 0;
  let scale = 1;
  let rotation = 0;

  function apiBase() {
    const configured = params.get("api");
    if (configured) return configured.replace(/\/$/, "");
    return detectedApiBase || window.location.origin;
  }

  async function detectApiBase() {
    if (params.get("api")) return apiBase();
    if (window.location.protocol !== "http:" || window.location.port !== "5173") {
      detectedApiBase = window.location.origin;
      return detectedApiBase;
    }
    try {
      const response = await fetch(`${window.location.origin}/api/v1/mobile/config?probe=${Date.now()}`, { cache: "no-store" });
      if (response.status !== 404 && response.status !== 501) {
        detectedApiBase = window.location.origin;
        return detectedApiBase;
      }
    } catch (_error) {
      // Static development server uses the API port below.
    }
    const hostname = window.location.hostname.includes(":") ? `[${window.location.hostname}]` : window.location.hostname;
    detectedApiBase = `http://${hostname}:8000`;
    return detectedApiBase;
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function fileUrl(source) {
    return `${apiBase()}/api/v1/applications/${encodeURIComponent(applicationNo)}`
      + `/files/${encodeURIComponent(source.material_code)}/${encodeURIComponent(source.name)}`;
  }

  function applyTransform() {
    viewerImage.style.transform = `scale(${scale}) rotate(${rotation}deg)`;
    const resetButton = viewer.querySelector('[data-viewer-action="reset"]');
    resetButton.textContent = `${Math.round(scale * 100)}%`;
  }

  function showCurrentImage() {
    const image = images[currentIndex];
    if (!image) return;
    scale = 1;
    rotation = 0;
    viewerImage.src = image.url;
    viewerImage.alt = `${image.label}：${image.name}`;
    viewerTitle.textContent = image.name;
    viewerPosition.textContent = `${image.label} · ${currentIndex + 1} / ${images.length}`;
    viewer.querySelector('[data-viewer-action="previous"]').disabled = images.length < 2;
    viewer.querySelector('[data-viewer-action="next"]').disabled = images.length < 2;
    viewerStage.scrollTo(0, 0);
    applyTransform();
  }

  function openViewer(index) {
    currentIndex = index;
    showCurrentImage();
    viewer.hidden = false;
    document.body.classList.add("viewer-open");
    viewer.querySelector('[data-viewer-action="close"]').focus();
  }

  function closeViewer() {
    viewer.hidden = true;
    viewerImage.removeAttribute("src");
    document.body.classList.remove("viewer-open");
  }

  function move(offset) {
    if (!images.length) return;
    currentIndex = (currentIndex + offset + images.length) % images.length;
    showCurrentImage();
  }

  function renderGroups(data) {
    images = [];
    groupsContainer.replaceChildren();
    data.groups.forEach((group) => {
      const section = element("section", "source-group");
      const header = element("header", "group-header");
      const title = element("div", "group-title");
      title.append(element("h2", "", group.label), element("span", "", group.material_code));
      header.append(title, element("span", "group-count", `${group.files.length} 张`));
      section.append(header);

      if (!group.files.length) {
        section.append(element("div", "empty-group", "该目录暂无图片"));
      } else {
        const grid = element("div", "image-grid");
        group.files.forEach((file) => {
          const imageIndex = images.length;
          const originalUrl = fileUrl(file);
          const source = {
            ...file,
            label: group.label,
            url: originalUrl,
            thumbnailUrl: `${originalUrl}?thumbnail=true`,
          };
          images.push(source);
          const button = element("button", "image-card");
          button.type = "button";
          button.title = `查看 ${file.name}`;
          const thumbnail = element("span", "thumbnail");
          const image = element("img");
          image.src = source.thumbnailUrl;
          image.alt = `${group.label}：${file.name}`;
          image.loading = "lazy";
          thumbnail.append(image);
          button.append(thumbnail, element("span", "file-name", file.name));
          button.addEventListener("click", () => openViewer(imageIndex));
          grid.append(button);
        });
        section.append(grid);
      }
      groupsContainer.append(section);
    });
    document.querySelector("#gallery-total").textContent = `${images.length} 张`;
    groupsContainer.hidden = false;
  }

  viewer.addEventListener("click", (event) => {
    const action = event.target.closest("[data-viewer-action]")?.dataset.viewerAction;
    if (!action) return;
    if (action === "close") closeViewer();
    if (action === "previous") move(-1);
    if (action === "next") move(1);
    if (action === "zoom-in") { scale = Math.min(5, scale + 0.25); applyTransform(); }
    if (action === "zoom-out") { scale = Math.max(0.25, scale - 0.25); applyTransform(); }
    if (action === "reset") { scale = 1; rotation = 0; applyTransform(); }
    if (action === "rotate-left") { rotation -= 90; applyTransform(); }
    if (action === "rotate-right") { rotation += 90; applyTransform(); }
  });

  viewerStage.addEventListener("wheel", (event) => {
    if (viewer.hidden) return;
    event.preventDefault();
    scale = Math.max(0.25, Math.min(5, scale + (event.deltaY < 0 ? 0.15 : -0.15)));
    applyTransform();
  }, { passive: false });

  document.addEventListener("keydown", (event) => {
    if (viewer.hidden) return;
    if (event.key === "Escape") closeViewer();
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
    if (["+", "="].includes(event.key)) { scale = Math.min(5, scale + 0.25); applyTransform(); }
    if (event.key === "-") { scale = Math.max(0.25, scale - 0.25); applyTransform(); }
  });

  async function load() {
    if (!/^\d{1,32}$/.test(applicationNo)) {
      loading.hidden = true;
      errorPanel.textContent = "缺少有效的申请单号";
      errorPanel.hidden = false;
      return;
    }
    document.querySelector("#gallery-title").textContent = `申请单 ${applicationNo} 原始图片`;
    const resultUrl = new URL("./application.html", window.location.href);
    resultUrl.search = "";
    resultUrl.searchParams.set("application_no", applicationNo);
    if (params.get("api")) resultUrl.searchParams.set("api", params.get("api"));
    document.querySelector("#back-to-result").href = resultUrl.toString();
    try {
      await detectApiBase();
      const response = await fetch(`${apiBase()}/api/v1/applications/${encodeURIComponent(applicationNo)}/files`, { cache: "no-store" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "原始图片加载失败");
      renderGroups(payload);
    } catch (error) {
      errorPanel.textContent = error instanceof Error ? error.message : "原始图片加载失败";
      errorPanel.hidden = false;
    } finally {
      loading.hidden = true;
    }
  }

  load();
})();
