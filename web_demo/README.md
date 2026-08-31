# Web Demo

这是面向手机浏览器的 OCR 接口测试页面，不是正式业务前端。

直接启动静态页面：

```powershell
python main.py
```

安装 Caddy 后，可通过同一地址访问页面和 API：

```powershell
python main.py --proxy --host 192.168.1.10 --api-upstream 127.0.0.1:8000
```

页面支持身份证正反面、驾驶证、行驶证和登记证拍照测试。
