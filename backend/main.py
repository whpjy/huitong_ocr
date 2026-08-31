"""Convenient entry point for starting the FastAPI development server.

Run this file from the ``backend`` directory with::

    python main.py
"""

from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("OCR_API_HOST", "0.0.0.0"),
        port=int(os.getenv("OCR_API_PORT", "8000")),
        reload=True,
    )
