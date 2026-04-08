"""Launch the webhook app on explicit IPv4 and IPv6 sockets."""

from __future__ import annotations

import asyncio
import socket

import uvicorn


HOST_V4 = "0.0.0.0"
HOST_V6 = "::"
PORT = 8001


def _build_socket(family: int, host: str) -> socket.socket:
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    sock.bind((host, PORT))
    sock.listen(socket.SOMAXCONN)
    sock.setblocking(False)
    return sock


async def _serve() -> None:
    config = uvicorn.Config(
        "paperless_ai.search.webhook:app",
        log_config="/app/uvicorn_logging.json",
    )
    server = uvicorn.Server(config)
    sockets = [
        _build_socket(socket.AF_INET, HOST_V4),
        _build_socket(socket.AF_INET6, HOST_V6),
    ]
    try:
        await server.serve(sockets=sockets)
    finally:
        for sock in sockets:
            sock.close()


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
