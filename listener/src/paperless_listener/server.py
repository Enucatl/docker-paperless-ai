"""Launch the service on explicit IPv4 and IPv6 sockets."""

import asyncio

from paperless_common.server import serve


def main() -> None:
    """Run the service with the shared dual-stack launcher."""
    asyncio.run(serve("paperless_listener.app:app"))


if __name__ == "__main__":
    main()
