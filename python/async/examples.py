"""Async examples."""

import asyncio


async def say_hello() -> None:
    await asyncio.sleep(0.1)
    print("hello async")


def main() -> None:
    asyncio.run(say_hello())


if __name__ == "__main__":
    main()
