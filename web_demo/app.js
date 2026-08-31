(() => {
  "use strict";

  // WeChat and some embedded WebViews keep an old stylesheet even after the
  // mounted static files change. app-loader already fetches this script with
  // a timestamp, so use the fresh script to refresh CSS as well.
  const stylesheet = document.querySelector('link[rel="stylesheet"]');
  if (stylesheet) {
    const stylesheetUrl = new URL(stylesheet.href, window.location.href);
    stylesheetUrl.searchParams.set("runtime", String(Date.now()));
    stylesheet.href = stylesheetUrl.href;
  }
  const viewport = document.querySelector('meta[name="viewport"]');
  if (viewport) {
    viewport.content = "width=device-width, initial-scale=1, minimum-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover";
  }

  const DOCUMENTS = {
    id_card: {
      name: "身份证",
      sides: { front: "人像面", back: "国徽面" },
      description: "请按顺序上传人像面和国徽面，确保边缘完整、文字清晰",
      fields: ["证件类型", "姓名", "身份证号", "出生日期", "性别", "民族", "住址", "签发机关", "有效期"],
    },
    driver_license: {
      name: "驾驶证",
      sides: { front: "证件图片", back: "" },
      description: "请拍摄或上传一张完整、文字清晰的驾驶证图片",
      fields: ["证件类型", "证号", "姓名", "性别", "国籍", "住址", "出生日期", "初次领证日期", "准驾车型", "有效期限", "档案编号"],
    },
    vehicle_license: {
      name: "行驶证",
      sides: { front: "证件图片", back: "" },
      description: "请拍摄或上传一张完整、文字清晰的行驶证图片",
      fields: ["号牌号码", "车辆类型", "所有人", "住址", "使用性质", "品牌型号", "车辆识别代号", "发动机号码", "注册日期", "发证日期", "档案编号", "核定载人数", "总质量", "整备质量", "外廓尺寸", "检验记录"],
    },
  };
  const SIDE_KEYS = ["front", "back"];
  // Keep the upload aligned with HunyuanOCR's configured vision input. Sending
  // a larger JPEG only adds browser encoding, network transfer, and a second
  // backend resize without giving the model any extra pixels.
  const MODEL_IMAGE_MAX_SIDE = 1920;
  // An ID-card request always contains two regular, tightly framed pages.
  // 1600px keeps its printed text clear while reducing each model's visual
  // encoding load by about 31% compared with two 1920px inputs.
  const ID_CARD_IMAGE_MAX_SIDE = 1600;
  const UPLOAD_JPEG_QUALITY = 0.9;
  const DIRECT_UPLOAD_IMAGE_TYPES = new Set([
    "image/jpeg",
    "image/png",
    "image/webp",
  ]);
  const emptySide = () => ({
    file: null,
    previewUrl: "",
    saveUrl: "",
    saveUrlExpiresAt: 0,
    quality: null,
    processing: false,
  });
  const state = {
    documentType: "id_card",
    sides: { front: emptySide(), back: emptySide() },
    captureSide: "front",
    pendingAlbumSide: "front",
    stream: null,
    fields: {},
  };

  const currentDocument = () => DOCUMENTS[state.documentType];
  const isTwoSidedDocument = () => state.documentType === "id_card";
  const requiredSides = () => isTwoSidedDocument() ? SIDE_KEYS : ["front"];
  const sideLabel = (side) => currentDocument().sides[side];

  const screens = [...document.querySelectorAll("[data-screen]")];
  const albumInput = document.querySelector("#album-input");
  const cameraFallbackInput = document.querySelector("#camera-fallback-input");
  const video = document.querySelector("#camera-video");
  const cameraCanvas = document.querySelector("#camera-canvas");
  const toast = document.querySelector("#toast");
  const imageViewer = document.querySelector("#image-viewer");
  const imageViewerStage = document.querySelector("#image-viewer-stage");
  const imageViewerImage = document.querySelector("#image-viewer-image");
  const imageViewerTitle = document.querySelector("#image-viewer-title");
  const imageViewerSaveTip = document.querySelector("#image-viewer-save-tip");
  const imageViewerZoom = document.querySelector("#image-viewer-zoom");
  let toastTimer = 0;
  let cameraQualityTimer = 0;
  let detectedApiBase = "";
  let viewerScale = 1;
  let viewerBaseWidth = 0;
  let viewerBaseHeight = 0;

  function showScreen(name) {
    screens.forEach((screen) => screen.classList.toggle("is-active", screen.dataset.screen === name));
    window.scrollTo(0, 0);
  }

  function waitForNextPaint() {
    return new Promise((resolve) => {
      window.requestAnimationFrame(() => window.requestAnimationFrame(resolve));
    });
  }

  function sizeImageViewer() {
    if (!imageViewerImage.naturalWidth || imageViewer.hidden) return;
    const availableWidth = Math.max(1, imageViewerStage.clientWidth - 24);
    const availableHeight = Math.max(1, imageViewerStage.clientHeight - 24);
    const fitScale = Math.min(
      availableWidth / imageViewerImage.naturalWidth,
      availableHeight / imageViewerImage.naturalHeight,
      1,
    );
    viewerBaseWidth = Math.max(1, Math.round(imageViewerImage.naturalWidth * fitScale));
    viewerBaseHeight = Math.max(1, Math.round(imageViewerImage.naturalHeight * fitScale));
    setImageViewerScale(1);
  }

  function setImageViewerScale(scale) {
    if (!viewerBaseWidth || !viewerBaseHeight) return;
    viewerScale = Math.min(4, Math.max(1, scale));
    imageViewerImage.style.width = `${Math.round(viewerBaseWidth * viewerScale)}px`;
    imageViewerImage.style.height = `${Math.round(viewerBaseHeight * viewerScale)}px`;
    imageViewerZoom.textContent = `${Math.round(viewerScale * 100)}%`;
    window.requestAnimationFrame(() => {
      imageViewerStage.scrollLeft = Math.max(
        0,
        (imageViewerStage.scrollWidth - imageViewerStage.clientWidth) / 2,
      );
      imageViewerStage.scrollTop = Math.max(
        0,
        (imageViewerStage.scrollHeight - imageViewerStage.clientHeight) / 2,
      );
    });
  }

  function openImageViewer(side, { saveMode = false, sourceUrl = "" } = {}) {
    const previewUrl = sourceUrl || state.sides[side]?.previewUrl;
    if (!previewUrl) return;
    imageViewer.dataset.saveMode = saveMode ? "true" : "false";
    imageViewerTitle.textContent = saveMode ? "查看大图" : "查看图片";
    imageViewerSaveTip.hidden = !saveMode;
    imageViewer.hidden = false;
    imageViewerImage.alt = `${currentDocument().name}${sideLabel(side)}大图`;
    imageViewerImage.src = previewUrl;
    if (imageViewerImage.complete) {
      window.requestAnimationFrame(sizeImageViewer);
    }
  }

  function closeImageViewer() {
    imageViewer.hidden = true;
    imageViewer.dataset.saveMode = "false";
    imageViewerTitle.textContent = "查看图片";
    imageViewerSaveTip.hidden = true;
    imageViewerImage.removeAttribute("src");
    imageViewerImage.style.removeProperty("width");
    imageViewerImage.style.removeProperty("height");
    viewerScale = 1;
    viewerBaseWidth = 0;
    viewerBaseHeight = 0;
    imageViewerZoom.textContent = "100%";
  }

  function showToast(message, duration = 2600) {
    window.clearTimeout(toastTimer);
    toast.textContent = message;
    toast.classList.add("is-visible");
    toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), duration);
  }

  function apiBase() {
    const fromQuery = new URLSearchParams(window.location.search).get("api");
    if (fromQuery) {
      const normalized = fromQuery.replace(/\/$/, "");
      // An old ?api=http://... bookmark must not break the HTTPS deployment.
      if (!(window.location.protocol === "https:" && normalized.startsWith("http://"))) {
        return normalized;
      }
    }
    if (detectedApiBase) return detectedApiBase;
    // The deployed Caddy server proxies /api through the same origin. Local
    // static-only mode is detected before recognition and switches to 8000.
    return window.location.origin;
  }

  async function detectApiBase() {
    const fromQuery = new URLSearchParams(window.location.search).get("api");
    if (fromQuery || detectedApiBase) return apiBase();
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
      // Keep the same-origin address when the proxy itself is unavailable so
      // the eventual error still points at the deployed service correctly.
      detectedApiBase = window.location.origin;
      return detectedApiBase;
    }

    const hostname = window.location.hostname.includes(":")
      ? `[${window.location.hostname}]`
      : window.location.hostname;
    detectedApiBase = `http://${hostname}:8000`;
    return detectedApiBase;
  }

  function firstMissingSide() {
    for (const side of requiredSides()) {
      if (!state.sides[side].file) return side;
    }
    return "front";
  }

  function bothSidesReady() {
    return requiredSides().every((side) => Boolean(state.sides[side].file));
  }

  function clearCapturedSides() {
    for (const side of SIDE_KEYS) {
      if (state.sides[side].previewUrl) URL.revokeObjectURL(state.sides[side].previewUrl);
      state.sides[side] = emptySide();
    }
    state.captureSide = "front";
    state.pendingAlbumSide = "front";
    state.fields = {};
  }

  function configureDocument(documentType) {
    if (!DOCUMENTS[documentType]) return;
    stopCamera();
    clearCapturedSides();
    state.documentType = documentType;
    const config = currentDocument();
    document.documentElement.dataset.document = documentType;
    document.querySelector("#method-title").textContent = isTwoSidedDocument()
      ? "完善身份证信息"
      : `${config.name}识别`;
    document.querySelector("#camera-document-title").textContent = `拍摄${config.name}`;
    document.querySelector("#success-description").textContent = `${config.name}信息已完成核对`;
    document.querySelectorAll("[data-two-sided-only]").forEach((element) => {
      element.hidden = !isTwoSidedDocument();
    });
    document.querySelectorAll("[data-id-card-only]").forEach((element) => {
      element.hidden = !isTwoSidedDocument();
    });
    document.querySelector(".side-upload-list").setAttribute(
      "aria-label",
      isTwoSidedDocument() ? "身份证双面上传" : `${config.name}单张上传`,
    );
    for (const side of SIDE_KEYS) {
      document.querySelector(`[data-side-label="${side}"]`).textContent = config.sides[side];
      document.querySelector(`[data-camera-label="${side}"]`).textContent = config.sides[side];
      document.querySelector(`[data-preview-label="${side}"]`).textContent = config.sides[side];
      document.querySelector(`[data-side-image="${side}"]`).alt = `${config.name}${config.sides[side]}预览`;
      document.querySelector(`[data-preview-image="${side}"]`).alt = `待识别的${config.name}${config.sides[side]}`;
    }
    renderMethodCards();
    showScreen("method");
  }

  function stopCamera() {
    window.clearInterval(cameraQualityTimer);
    cameraQualityTimer = 0;
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    video.srcObject = null;
  }

  async function openCamera(side = firstMissingSide()) {
    state.captureSide = side;
    updateCameraUi();
    if (!navigator.mediaDevices?.getUserMedia) {
      state.pendingAlbumSide = side;
      showToast("当前 HTTP 页面无法使用实时相机，将打开系统拍照");
      cameraFallbackInput.click();
      return;
    }
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 3840 },
          height: { ideal: 2160 },
          aspectRatio: { ideal: 16 / 9 },
        },
      });
      await preferContinuousFocus(state.stream);
      video.srcObject = state.stream;
      showScreen("camera");
      await video.play();
      startCameraHints();
    } catch (error) {
      stopCamera();
      state.pendingAlbumSide = side;
      showToast(error?.name === "NotAllowedError" ? "未获得实时相机权限，将打开系统拍照" : "无法打开实时相机，将打开系统拍照");
      cameraFallbackInput.click();
    }
  }

  async function preferContinuousFocus(stream) {
    const track = stream.getVideoTracks()[0];
    if (!track?.getCapabilities || !track.applyConstraints) return;
    try {
      const capabilities = track.getCapabilities();
      if (capabilities.focusMode?.includes("continuous")) {
        await track.applyConstraints({ advanced: [{ focusMode: "continuous" }] });
      }
    } catch (_) {
      // Some mobile WebViews expose focus capabilities but reject applying them.
    }
  }

  function updateCameraUi() {
    const side = state.captureSide;
    document.querySelector("#camera-side-title").textContent = `请扫描${currentDocument().name}${sideLabel(side)}`;
    const isIdCard = state.documentType === "id_card";
    document.querySelector("#guide-ghost").className = `guide-ghost ${isIdCard ? (side === "front" ? "portrait-ghost" : "emblem-ghost") : (side === "front" ? "document-front-ghost" : "document-back-ghost")}`;
    document.querySelectorAll("[data-camera-side]").forEach((item) => {
      const itemSide = item.dataset.cameraSide;
      item.classList.toggle("is-active", itemSide === side);
      item.classList.toggle("is-done", Boolean(state.sides[itemSide].file));
      item.querySelector(".mini-card").style.backgroundImage = state.sides[itemSide].previewUrl ? `url("${state.sides[itemSide].previewUrl}")` : "";
    });
    const hint = document.querySelector("#camera-hint");
    hint.textContent = `请靠近拍摄，使${currentDocument().name}边缘贴近取景框`;
    hint.classList.remove("is-warning");
  }

  function startCameraHints() {
    const hint = document.querySelector("#camera-hint");
    window.clearInterval(cameraQualityTimer);
    cameraQualityTimer = window.setInterval(() => {
      if (!video.videoWidth) return;
      const light = getLightStats(sampleVideoFrame(240).data);
      const defaultMessage = `请靠近拍摄，使${currentDocument().name}边缘贴近取景框`;
      let message = defaultMessage;
      if (light.mean < 48) message = "光线较暗，请移到明亮处";
      else if (light.mean > 225) message = "画面过亮，请避免灯光直射";
      hint.textContent = message;
      hint.classList.toggle("is-warning", message !== defaultMessage);
    }, 500);
  }

  function sampleVideoFrame(targetWidth) {
    const scale = targetWidth / video.videoWidth;
    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = targetWidth;
    sampleCanvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = sampleCanvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(video, 0, 0, sampleCanvas.width, sampleCanvas.height);
    return context.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height);
  }

  async function capturePhoto() {
    if (!video.videoWidth) return showToast("相机还未准备好，请稍候");
    const crop = videoGuideCrop();
    cameraCanvas.width = crop.width;
    cameraCanvas.height = crop.height;
    cameraCanvas.getContext("2d", { alpha: false }).drawImage(
      video,
      crop.x,
      crop.y,
      crop.width,
      crop.height,
      0,
      0,
      crop.width,
      crop.height,
    );
    const quality = analyzeCanvas(cameraCanvas, crop.width, crop.height);
    const uploadCanvas = resizeCanvasForModel(cameraCanvas);
    const blob = await new Promise((resolve) => uploadCanvas.toBlob(
      resolve,
      "image/jpeg",
      UPLOAD_JPEG_QUALITY,
    ));
    if (!blob) return showToast("拍照失败，请重试");
    const capturedSide = state.captureSide;
    const capturedFile = new File(
      [blob],
      `${state.documentType}-${capturedSide}-${Date.now()}.jpg`,
      { type: "image/jpeg" },
    );
    const stored = storePreparedSideFile(capturedSide, capturedFile, quality);
    if (!stored) return;
    if (isTwoSidedDocument() && capturedSide === "front" && !state.sides.back.file) {
      state.captureSide = "back";
      updateCameraUi();
      showToast(`${sideLabel("front")}已拍好，请继续拍摄${sideLabel("back")}`);
      return;
    }
    stopCamera();
    renderPreview();
    showScreen("preview");
  }

  function videoGuideCrop() {
    const videoRect = video.getBoundingClientRect();
    const guideRect = document.querySelector(".guide-frame").getBoundingClientRect();
    const coverScale = Math.max(
      videoRect.width / video.videoWidth,
      videoRect.height / video.videoHeight,
    );
    const renderedLeft = videoRect.left + (videoRect.width - video.videoWidth * coverScale) / 2;
    const renderedTop = videoRect.top + (videoRect.height - video.videoHeight * coverScale) / 2;
    const left = Math.max(0, (guideRect.left - renderedLeft) / coverScale);
    const top = Math.max(0, (guideRect.top - renderedTop) / coverScale);
    const right = Math.min(video.videoWidth, (guideRect.right - renderedLeft) / coverScale);
    const bottom = Math.min(video.videoHeight, (guideRect.bottom - renderedTop) / coverScale);
    return {
      x: Math.round(left),
      y: Math.round(top),
      width: Math.max(1, Math.round(right - left)),
      height: Math.max(1, Math.round(bottom - top)),
    };
  }

  function chooseFromAlbum(side = firstMissingSide()) {
    state.pendingAlbumSide = side;
    albumInput.click();
  }

  async function viewSideImage(side) {
    const file = state.sides[side]?.file;
    if (!file) {
      showToast("请先选择图片");
      return;
    }

    try {
      showToast("正在准备大图…", 5000);
      const sideState = state.sides[side];
      if (!sideState.saveUrl || Date.now() >= sideState.saveUrlExpiresAt) {
        await detectApiBase();
        const data = new FormData();
        data.append("file", file, file.name);
        const response = await fetch(`${apiBase()}/api/v1/mobile/save-image`, {
          method: "POST",
          body: data,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload.url) {
          throw new Error(payload.detail || "服务器未能生成图片地址");
        }
        sideState.saveUrl = new URL(payload.url, `${apiBase()}/`).href;
        sideState.saveUrlExpiresAt = Date.now() + Number(payload.expires_in || 600) * 1000 - 5000;
      }
      openImageViewer(side, { saveMode: true, sourceUrl: sideState.saveUrl });
      showToast("可缩放查看，长按图片可以保存", 3200);
    } catch (error) {
      openImageViewer(side);
      showToast(error.message || "可保存大图准备失败", 4200);
    }
  }

  async function storeSideFile(side, file) {
    if (!file) return false;
    if (!file.type.startsWith("image/")) return showToast("请选择图片文件"), false;
    if (file.size > 15 * 1024 * 1024) return showToast("图片不能超过 15MB"), false;
    // The backend already normalizes orientation, limits model resolution and
    // performs the authoritative quality gate. Avoid decoding and re-encoding
    // regular phone images on the WebView main thread.
    const directUpload = DIRECT_UPLOAD_IMAGE_TYPES.has(file.type.toLowerCase())
      || /\.(?:jpe?g|png|webp)$/i.test(file.name);
    if (directUpload) {
      return storePreparedSideFile(side, file, {
        issues: [],
        deferredToServer: true,
      });
    }
    const previousSide = showPendingSidePreview(side, file);
    await waitForNextPaint();
    try {
      const prepared = await prepareImage(file);
      if (previousSide.previewUrl) URL.revokeObjectURL(previousSide.previewUrl);
      return storePreparedSideFile(side, prepared.file, prepared.quality);
    } catch (error) {
      restoreSidePreview(side, previousSide);
      showToast(error.message || "图片读取失败，请重新选择");
      return false;
    }
  }

  function showPendingSidePreview(side, file) {
    const previousSide = state.sides[side];
    state.sides[side] = {
      file: null,
      quality: null,
      previewUrl: URL.createObjectURL(file),
      saveUrl: "",
      saveUrlExpiresAt: 0,
      processing: true,
    };
    renderMethodCards();
    return previousSide;
  }

  function restoreSidePreview(side, previousSide) {
    if (state.sides[side].previewUrl) URL.revokeObjectURL(state.sides[side].previewUrl);
    state.sides[side] = previousSide;
    renderMethodCards();
  }

  function storePreparedSideFile(side, file, quality) {
    if (state.sides[side].previewUrl) URL.revokeObjectURL(state.sides[side].previewUrl);
    state.sides[side] = {
      file,
      quality,
      previewUrl: URL.createObjectURL(file),
      saveUrl: "",
      saveUrlExpiresAt: 0,
      processing: false,
    };
    renderMethodCards();
    return true;
  }

  function renderMethodCards() {
    for (const side of requiredSides()) {
      const card = document.querySelector(`[data-action="select-side"][data-side="${side}"]`);
      const image = document.querySelector(`[data-side-image="${side}"]`);
      const ready = Boolean(state.sides[side].file);
      const hasPreview = Boolean(state.sides[side].previewUrl);
      const processing = Boolean(state.sides[side].processing);
      const viewButton = card.querySelector(".view-image-button");
      if (viewButton) viewButton.hidden = !ready;
      card.classList.toggle("has-image", hasPreview);
      card.classList.toggle("is-processing", processing);
      document.querySelector(`[data-side-status="${side}"]`).textContent = processing
        ? "正在处理…"
        : ready
          ? "已上传 · 点击更换"
          : "点击上传";
      if (hasPreview) image.src = state.sides[side].previewUrl;
      else image.removeAttribute("src");
      const prompt = document.querySelector(`[data-upload-prompt="${side}"]`);
      if (prompt) prompt.textContent = ready
        ? `${sideLabel(side)}已上传，点击更换`
        : `点击上传${sideLabel(side)}`;
    }
    const submitButton = document.querySelector("#document-submit-button");
    if (submitButton) submitButton.disabled = !bothSidesReady();
  }

  function renderPreview() {
    let allGood = true;
    for (const side of requiredSides()) {
      const data = state.sides[side];
      document.querySelector(`[data-preview-image="${side}"]`).src = data.previewUrl;
      const qualityText = document.querySelector(`[data-quality-text="${side}"]`);
      const issues = data.quality?.issues || [];
      qualityText.textContent = issues.length
        ? issues.join("、")
        : data.quality?.deferredToServer
          ? "图片已选择，提交后由服务器检测质量"
          : `${data.quality.width} × ${data.quality.height}，质量良好`;
      qualityText.classList.toggle("is-warning", Boolean(issues.length));
      if (issues.length) allGood = false;
    }
    const card = document.querySelector("#quality-card");
    card.classList.toggle("is-warning", !allGood);
    card.classList.remove("is-error");
    card.querySelector(".quality-icon").textContent = allGood ? "✓" : "!";
    document.querySelector("#quality-title").textContent = allGood
      ? (isTwoSidedDocument() ? "两面照片已准备好" : "照片已准备好")
      : "照片可能影响识别";
    document.querySelector("#quality-detail").textContent = allGood
      ? (isTwoSidedDocument() ? "将识别并合并正反面字段" : "将使用双模型并行识别")
      : "可以重新拍摄，或继续尝试识别";
  }

  async function prepareImage(file) {
    const image = await loadImage(file);
    const maxSide = uploadImageMaxSide();
    const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
    context.fillStyle = "#fff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const quality = analyzeCanvas(canvas, image.naturalWidth, image.naturalHeight);
    const blob = await new Promise((resolve) => canvas.toBlob(
      resolve,
      "image/jpeg",
      UPLOAD_JPEG_QUALITY,
    ));
    if (!blob) throw new Error("当前浏览器无法处理这张图片");
    return { file: new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" }), quality };
  }

  function resizeCanvasForModel(source) {
    const scale = Math.min(1, uploadImageMaxSide() / Math.max(source.width, source.height));
    if (scale === 1) return source;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(source.width * scale));
    canvas.height = Math.max(1, Math.round(source.height * scale));
    const context = canvas.getContext("2d", { alpha: false });
    context.fillStyle = "#fff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  function uploadImageMaxSide() {
    return state.documentType === "id_card"
      ? ID_CARD_IMAGE_MAX_SIDE
      : MODEL_IMAGE_MAX_SIDE;
  }

  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
      image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("暂不支持该图片格式，请使用 JPG 或 PNG")); };
      image.src = url;
    });
  }

  function analyzeCanvas(source, originalWidth, originalHeight) {
    const width = 320;
    const height = Math.max(1, Math.round(source.height * width / source.width));
    const sample = document.createElement("canvas");
    sample.width = width;
    sample.height = height;
    const context = sample.getContext("2d", { willReadFrequently: true });
    context.drawImage(source, 0, 0, width, height);
    const imageData = context.getImageData(0, 0, width, height);
    const light = getLightStats(imageData.data);
    const sharpness = getSharpness(imageData.data, width, height);
    const issues = [];
    if (Math.min(originalWidth, originalHeight) < 600) issues.push("分辨率偏低");
    if (light.mean < 45) issues.push("画面过暗");
    if (light.mean > 228) issues.push("画面过亮或反光");
    if (light.contrast < 24) issues.push("对比度偏低");
    if (sharpness < 65) issues.push("图片可能模糊");
    return { issues, width: originalWidth, height: originalHeight, light, sharpness };
  }

  function getLightStats(data) {
    let sum = 0, sumSquare = 0, count = 0;
    for (let i = 0; i < data.length; i += 16) {
      const value = .299 * data[i] + .587 * data[i + 1] + .114 * data[i + 2];
      sum += value; sumSquare += value * value; count += 1;
    }
    const mean = sum / Math.max(1, count);
    return { mean, contrast: Math.sqrt(Math.max(0, sumSquare / Math.max(1, count) - mean * mean)) };
  }

  function getSharpness(data, width, height) {
    let total = 0, count = 0;
    const gray = (index) => .299 * data[index] + .587 * data[index + 1] + .114 * data[index + 2];
    for (let y = 2; y < height - 2; y += 2) {
      for (let x = 2; x < width - 2; x += 2) {
        const i = (y * width + x) * 4;
        const laplacian = 4 * gray(i) - gray(i - 4) - gray(i + 4) - gray(i - width * 4) - gray(i + width * 4);
        total += laplacian * laplacian;
        count += 1;
      }
    }
    return total / Math.max(1, count);
  }

  function qualityRejectionMessage(detail) {
    const issues = Array.isArray(detail?.issues) ? detail.issues : [];
    const messages = issues
      .filter((issue) => issue && issue.message)
      .map((issue) => {
        const label = issue.side_label;
        return label && label !== "证件图片"
          ? `${label}：${issue.message}`
          : issue.message;
      });
    return [...new Set(messages)].join("；");
  }

  function recognitionError(payload, status, fallbackMessage) {
    const detail = payload?.detail;
    if (detail && typeof detail === "object") {
      const isQualityRejection = detail.code === "IMAGE_QUALITY_REJECTED"
        || detail.code === "IMAGE_QUALITY_CHECK_FAILED";
      const error = new Error(
        (isQualityRejection && qualityRejectionMessage(detail))
        || detail.message
        || fallbackMessage,
      );
      if (isQualityRejection) {
        error.qualityRejection = detail;
      }
      return error;
    }
    return new Error(`${fallbackMessage}：${detail || payload?.message || status}`);
  }

  function renderServerQualityRejection(detail) {
    const issues = Array.isArray(detail?.issues) ? detail.issues : [];
    const sideMap = { DG12: "front", DG13: "back", document: "front" };
    for (const side of requiredSides()) {
      const messages = issues
        .filter((issue) => sideMap[issue.side] === side)
        .map((issue) => issue.message)
        .filter(Boolean);
      if (!messages.length) continue;
      const qualityText = document.querySelector(`[data-quality-text="${side}"]`);
      qualityText.textContent = messages.join("；");
      qualityText.classList.add("is-warning");
    }
    const card = document.querySelector("#quality-card");
    card.classList.remove("is-warning");
    card.classList.add("is-error");
    card.querySelector(".quality-icon").textContent = "×";
    document.querySelector("#quality-title").textContent = "照片质量不合格，请重新拍摄";
    document.querySelector("#quality-detail").textContent = qualityRejectionMessage(detail)
      || detail.message
      || "图片质量检测失败";
  }

  async function recognizeSide(side) {
    document.querySelector("#loading-detail").textContent = `正在识别${sideLabel(side)}…`;
    const data = new FormData();
    data.append("file", state.sides[side].file, state.sides[side].file.name);
    data.append("document_type", state.documentType);
    const response = await fetch(`${apiBase()}/api/v1/recognition/document`, { method: "POST", body: data });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw recognitionError(payload, response.status, `${sideLabel(side)}识别失败`);
    return payload;
  }

  async function recognizeIdCard() {
    document.querySelector("#loading-detail").textContent = "正在并行识别身份证正反面…";
    const data = new FormData();
    data.append("front_file", state.sides.front.file, state.sides.front.file.name);
    data.append("back_file", state.sides.back.file, state.sides.back.file.name);
    data.append("document_type", state.documentType);
    const response = await fetch(`${apiBase()}/api/v1/recognition/document`, { method: "POST", body: data });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw recognitionError(payload, response.status, "身份证识别失败");
    return payload;
  }

  async function recognize(failureScreen = "preview") {
    if (!bothSidesReady()) {
      return showToast(isTwoSidedDocument()
        ? `请先上传${currentDocument().name}${sideLabel("front")}和${sideLabel("back")}`
        : `请先上传${currentDocument().name}图片`);
    }
    showScreen("loading");
    // Some mobile WebViews defer rendering while JavaScript immediately
    // starts a multipart upload. Let the loading screen reach the compositor
    // before API probing and recognition begin.
    await waitForNextPaint();
    const wallStarted = performance.now();
    try {
      await detectApiBase();
      let result;
      if (state.documentType === "id_card") {
        result = await recognizeIdCard();
      } else {
        result = await recognizeSide("front");
      }
      state.fields = { ...(result.fields || {}) };
      if (state.documentType === "vehicle_license" && !state.fields["档案编号"]) {
        state.fields["档案编号"] = state.fields["档案号码"] || "";
      }
      renderFields(state.fields);
      const wallSeconds = (performance.now() - wallStarted) / 1000;
      const backendSeconds = Number(result.timing?.total_seconds);
      if (Number.isFinite(backendSeconds) && backendSeconds >= 0) {
        const uploadSeconds = Math.max(0, wallSeconds - backendSeconds);
        document.querySelector("#result-meta").textContent =
          `上传耗时 ${uploadSeconds.toFixed(2)} 秒 · 模型处理耗时 ${backendSeconds.toFixed(2)} 秒`;
      } else {
        document.querySelector("#result-meta").textContent =
          `模型处理耗时 ${wallSeconds.toFixed(2)} 秒`;
      }
      showScreen("result");
    } catch (error) {
      showScreen(failureScreen);
      if (error.qualityRejection) renderServerQualityRejection(error.qualityRejection);
      const hint = error instanceof TypeError
        ? `无法连接识别服务（${apiBase()}）。请确认 OCR 服务容器和 /api 代理正常运行`
        : error.message;
      showToast(hint || "识别失败，请稍后重试", error.qualityRejection ? 6500 : 4200);
    }
  }

  function renderFields(fields) {
    const form = document.querySelector("#result-form");
    const order = currentDocument().fields;
    const names = [...order, ...Object.keys(fields).filter((name) => !order.includes(name))];
    form.replaceChildren(...names.map((name, index) => {
      const wrapper = document.createElement("div");
      wrapper.className = "form-field";
      const id = `field-${index}`;
      const label = document.createElement("label");
      label.htmlFor = id;
      label.textContent = name;
      const multiline = name.includes("住址") || String(fields[name] || "").length > 30;
      const input = document.createElement(multiline ? "textarea" : "input");
      input.id = id;
      input.name = name;
      input.value = fields[name] || "";
      input.placeholder = "未识别，请手动填写";
      if (name.includes("身份证号")) input.autocapitalize = "characters";
      wrapper.append(label, input);
      return wrapper;
    }));
  }

  function reset() {
    stopCamera();
    clearCapturedSides();
    document.documentElement.removeAttribute("data-document");
    renderMethodCards();
    showScreen("home");
  }

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    if (action === "open-document") configureDocument(button.dataset.document);
    else if (action === "view-image") await viewSideImage(button.dataset.side);
    else if (action === "close-image-viewer") closeImageViewer();
    else if (action === "zoom-in") setImageViewerScale(viewerScale + 0.5);
    else if (action === "zoom-out") setImageViewerScale(viewerScale - 0.5);
    else if (action === "coming-soon") showToast(`${button.dataset.name}识别将在下一版本开放`);
    else if (action === "home" || action === "restart") reset();
    else if (action === "select-side") chooseFromAlbum(button.dataset.side);
    else if (action === "album") chooseFromAlbum(state.stream ? state.captureSide : firstMissingSide());
    else if (action === "camera") await openCamera(firstMissingSide());
    else if (action === "close-camera") { stopCamera(); showScreen("method"); }
    else if (action === "capture") await capturePhoto();
    else if (action === "replace-side") await openCamera(button.dataset.side);
    else if (action === "back-to-method" || action === "retake") showScreen("method");
    else if (action === "recognize") await recognize();
    else if (action === "submit-document") await recognize("method");
  });

  imageViewerImage.addEventListener("load", sizeImageViewer);
  imageViewerImage.addEventListener("dblclick", () => {
    setImageViewerScale(viewerScale === 1 ? 2 : 1);
  });
  window.addEventListener("resize", () => {
    if (!imageViewer.hidden) sizeImageViewer();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !imageViewer.hidden) closeImageViewer();
    if ((event.key === "Enter" || event.key === " ") && event.target.matches("[data-action='select-side']")) {
      event.preventDefault();
      chooseFromAlbum(event.target.dataset.side);
    }
  });

  albumInput.addEventListener("change", async () => {
    const selected = albumInput.files?.[0];
    albumInput.value = "";
    if (!selected) return;
    const side = state.pendingAlbumSide;
    const stored = await storeSideFile(side, selected);
    if (!stored) return;
    if (state.stream) stopCamera();
    if (bothSidesReady()) {
      showScreen("method");
    } else {
      showScreen("method");
      showToast(`${sideLabel(side)}已上传，请继续上传${sideLabel(firstMissingSide())}`);
    }
  });

  cameraFallbackInput.addEventListener("change", async () => {
    const selected = cameraFallbackInput.files?.[0];
    cameraFallbackInput.value = "";
    if (!selected) return;
    const side = state.pendingAlbumSide;
    const stored = await storeSideFile(side, selected);
    if (!stored) return;
    if (bothSidesReady()) {
      showScreen("method");
      showToast(`${currentDocument().name}图片已拍摄，可以提交识别`);
    } else {
      showScreen("method");
      showToast(`${sideLabel(side)}已拍好，请点击“开始拍照”继续拍摄${sideLabel(firstMissingSide())}`);
    }
  });

  document.querySelector("#result-form").addEventListener("submit", (event) => {
    event.preventDefault();
    state.fields = Object.fromEntries(new FormData(event.currentTarget).entries());
    showScreen("success");
  });
  window.addEventListener("pagehide", stopCamera);

  const requestedDocument = new URLSearchParams(window.location.search).get("document");
  if (DOCUMENTS[requestedDocument]) configureDocument(requestedDocument);
})();
