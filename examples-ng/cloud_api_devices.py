"""
List available cloud devices using TyxonQ cloud API facade.

- Reads token from env TYXONQ_API_KEY or uses existing configured token
- Prints device ids with provider prefix
"""

from __future__ import annotations

import os
import json
import tyxonq as tq
import getpass


def main():
    token = getpass.getpass("Enter your token: ")
    if token:
        tq.set_token(token, provider="tyxonq", device=None)
    devs = tq.api.list_devices(provider="tyxonq") if hasattr(tq, "api") else []
    print(json.dumps(devs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()



