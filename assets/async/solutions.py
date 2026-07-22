"""Async solutions."""

import asyncio


async def task(name: str) -> str:
    await asyncio.sleep(0.05)
    return name


def main() -> None:
    print(asyncio.run(task("done")))


if __name__ == "__main__":
    main()
