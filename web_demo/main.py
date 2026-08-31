"""Serve the mobile web app with Python's built-in HTTP server."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


WEB_DEMO_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动移动端样式的 OCR 测试页面")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认：0.0.0.0）")
    parser.add_argument("--port", type=int, default=5173, help="监听端口（默认：5173）")
    parser.add_argument(
        "--proxy",
        "--https",
        dest="proxy",
        action="store_true",
        help="通过 Caddy 启动 HTTP，并将 /api 代理到 OCR 后端",
    )
    parser.add_argument(
        "--api-upstream",
        default="127.0.0.1:8000",
        help="Caddy 代理的 OCR 后端地址（默认：127.0.0.1:8000）",
    )
    return parser.parse_args()


def run_proxy(args: argparse.Namespace) -> None:
    if args.host in {"0.0.0.0", "::"}:
        raise SystemExit(
            "Caddy 代理模式必须用 --host 指定手机访问的 IP 或域名，"
            "例如：python main.py --proxy --host 192.168.1.10"
        )

    caddy = shutil.which("caddy")
    if not caddy:
        raise SystemExit(
            "未找到 Caddy。请先安装 Caddy 并确认 caddy 命令已加入 PATH。"
        )

    env = os.environ.copy()
    env["MOBILE_HTTP_HOST"] = args.host
    env["MOBILE_HTTP_PORT"] = str(args.port)
    env["OCR_API_UPSTREAM"] = args.api_upstream

    print(f"移动端 HTTP 已启动：http://{args.host}:{args.port}")
    print(f"识别接口代理到：http://{args.api_upstream}")
    print("按 Ctrl+C 停止服务")
    try:
        subprocess.run(
            [caddy, "run", "--config", str(WEB_DEMO_DIR / "Caddyfile")],
            cwd=WEB_DEMO_DIR,
            env=env,
            check=True,
        )
    except KeyboardInterrupt:
        print("\n移动端 HTTP 代理已停止")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Caddy 启动失败，退出码：{exc.returncode}") from exc


def run_http(args: argparse.Namespace) -> None:

    def handler(*handler_args, **handler_kwargs):
        return SimpleHTTPRequestHandler(
            *handler_args,
            directory=str(WEB_DEMO_DIR),
            **handler_kwargs,
        )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"OCR 测试页面已启动：http://127.0.0.1:{args.port}")
    print(f"手机请访问：http://电脑局域网IP:{args.port}")
    print("按 Ctrl+C 停止服务")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n移动端已停止")
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    if args.proxy:
        run_proxy(args)
    else:
        run_http(args)


if __name__ == "__main__":
    main()
