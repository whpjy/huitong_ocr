(() => {
  "use strict";

  const script = document.createElement("script");
  script.src = `./app.js?runtime=${Date.now()}`;
  script.async = false;
  script.onerror = () => {
    const toast = document.querySelector("#toast");
    if (!toast) return;
    toast.textContent = "页面脚本加载失败，请关闭页面后重新打开";
    toast.classList.add("is-visible");
  };
  document.body.append(script);
})();
