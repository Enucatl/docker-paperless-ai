import os
import sys
import time

import niquests


def main() -> int:
    url = os.environ.get("PAPERLESS_URL", "http://webserver:8000")
    payload = {
        "username": os.environ.get("TEST_PAPERLESS_USER", "admin"),
        "password": os.environ.get("TEST_PAPERLESS_PASS", "admin"),
    }

    for _ in range(40):
        try:
            response = niquests.post(f"{url}/api/token/", json=payload, timeout=15)
            if response.status_code == 200:
                print(response.json()["token"])
                return 0
        except Exception:
            pass
        time.sleep(3)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
