"""Shared server startup preserves both sockets and service entrypoints."""

import socket
from unittest.mock import AsyncMock, Mock

import pytest

from paperless_common import server


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_dual_stack_socket_options_and_cleanup(monkeypatch, fails):
    """Both bound sockets close after normal serving or a server error."""
    sockets = [Mock(), Mock()]
    socket_factory = Mock(side_effect=sockets)
    monkeypatch.setattr(server.socket, "socket", socket_factory)
    config = Mock()
    instance = Mock(
        serve=AsyncMock(side_effect=RuntimeError("stopped") if fails else None)
    )
    monkeypatch.setattr(server.uvicorn, "Config", config)
    monkeypatch.setattr(server.uvicorn, "Server", Mock(return_value=instance))
    if fails:
        with pytest.raises(RuntimeError, match="stopped"):
            await server.serve("service:app")
    else:
        await server.serve("service:app")
    config.assert_called_once_with(
        "service:app", log_config="/app/uvicorn_logging.json"
    )
    instance.serve.assert_awaited_once_with(sockets=sockets)
    assert [call.args for call in socket_factory.call_args_list] == [
        (socket.AF_INET, socket.SOCK_STREAM),
        (socket.AF_INET6, socket.SOCK_STREAM),
    ]
    for sock, host in zip(sockets, ["0.0.0.0", "::"]):
        sock.bind.assert_called_once_with((host, 8001))
        sock.listen.assert_called_once_with(socket.SOMAXCONN)
        sock.setblocking.assert_called_once_with(False)
        sock.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.close.assert_called_once()
    assert sockets[0].setsockopt.call_count == 1
    sockets[1].setsockopt.assert_any_call(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)


@pytest.mark.parametrize(
    "module,app_path",
    [
        ("paperless_listener.server", "paperless_listener.app:app"),
        ("paperless_ai.search.copilot_server", "paperless_ai.search.webhook:app"),
    ],
)
def test_entrypoint_selects_service(monkeypatch, module, app_path):
    """Each entrypoint invokes the shared launcher with its own application."""
    import importlib

    entrypoint = importlib.import_module(module)
    serve = AsyncMock()
    monkeypatch.setattr(entrypoint, "serve", serve)
    entrypoint.main()
    serve.assert_awaited_once_with(app_path)
