"""API examples.

Install requests if needed: pip install requests
"""

import requests


def main() -> None:
    response = requests.get("https://httpbin.org/get", timeout=10)
    print(response.status_code)


if __name__ == "__main__":
    main()
