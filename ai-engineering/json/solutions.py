"""JSON solutions."""

import json


def main() -> None:
    payload = {"ok": True}
    encoded = json.dumps(payload)
    print(json.loads(encoded))


if __name__ == "__main__":
    main()
