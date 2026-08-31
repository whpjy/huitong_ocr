# huitong_ocr

从 `huitong_ocr_extract` 独立出的轻量证件 OCR 服务，仅保留移动端样式测试页面及其后端识别链路。

## 保留能力

- HunyuanOCR 图片结构化识别
- PP-OCRv6
- HunyuanOCR + PP-OCRv6 混合校验
- 图片方向纠正、证件裁边和质量检测
- 身份证、驾驶证、行驶证、登记证
- FastAPI、Web Demo 和 Docker 部署

本项目不包含模型评测、历史任务、Excel 批处理、管理前端以及其他模型 Provider。

## 目录

```text
huitong_ocr/
├── backend/          # 精简 FastAPI 识别服务
├── web_demo/         # 面向手机浏览器的手工测试页面
├── container-entrypoint.sh
├── Dockerfile
└── start.bat
```

## 配置

服务默认地址位于 `backend/config/ocr.yaml` 和 `backend/config/multimodal.yaml`，也可以通过环境变量覆盖：

```powershell
$env:HUNYUAN_OCR_BASE_URL = "http://你的混元服务地址/v1"
$env:HUNYUAN_OCR_API_KEY = "服务需要的密钥"
$env:PPOCR_BASE_URL = "http://你的PP-OCR服务地址"
```

仓库不保存真实 API Key。可参考 `.env.example`，但 Python 进程不会自动读取 `.env` 文件。

识别模式在 `backend/config/mobile.yaml` 中选择：`multimodal:hunyuan_ocr` 或 `hybrid:hunyuan_ocr`，必须且只能启用一个。

## 本地运行

安装依赖：

```powershell
cd backend
python -m pip install -r requirements.txt
```

分别启动后端和测试页面：

```powershell
cd backend
python main.py

cd web_demo
python main.py
```

也可以在 Windows 双击 `start.bat`。后端文档地址为 `http://127.0.0.1:8000/docs`，测试页面为 `http://127.0.0.1:5173`。

## Docker

```powershell
docker build -t huitong-ocr .
docker run --rm -p 5173:5173 `
  -e HUNYUAN_OCR_BASE_URL=http://混元服务地址/v1 `
  -e HUNYUAN_OCR_API_KEY=你的密钥 `
  -e PPOCR_BASE_URL=http://PP-OCR服务地址 `
  huitong-ocr
```

容器只对外开放 `5173`，Caddy 提供测试页面并将 `/api/*` 转发给容器内的 FastAPI。

## 测试

```powershell
cd backend
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
