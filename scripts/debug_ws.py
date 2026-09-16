#!/usr/bin/env python3
"""
Изолированная диагностика WebSocket Binance Futures.
Показывает, какие потоки реально приходят от биржи.

Запуск:
    python scripts/debug_ws.py
"""
import asyncio
import json
import websockets


async def test_combined() -> None:
    """Тест combined stream — точно такой же URL, как в нашем коде."""
    url = (
        "wss://fstream.binance.com/stream?streams="
        "btcusdt@bookTicker/btcusdt@aggTrade/btcusdt@depth@100ms"
    )

    print("=" * 60)
    print("TEST 1: COMBINED STREAM")
    print(f"URL: {url}")
    print("=" * 60)

    counts: dict[str, int] = {}

    async with websockets.connect(
        url,
        compression=None,
        open_timeout=10,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        print("Connected! Collecting 300 messages...\n")

        for i in range(300):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
            except asyncio.TimeoutError:
                print(f"Timeout after {i} messages")
                break

            data = json.loads(msg)
            stream = data.get("stream", "NO_STREAM")
            counts[stream] = counts.get(stream, 0) + 1

            # Печатаем первые 10 сообщений полностью
            if i < 10:
                print(f"  #{i + 1:>3}  stream={stream}")
                print(f"         raw: {msg[:250]}")
                print()

    print(f"\nStream counts ({sum(counts.values())} total):")
    for stream, count in sorted(counts.items()):
        print(f"  {stream}: {count}")


async def test_single_aggtrade() -> None:
    """Тест отдельного aggTrade потока (не combined)."""
    url = "wss://fstream.binance.com/ws/btcusdt@aggTrade"

    print()
    print("=" * 60)
    print("TEST 2: SINGLE aggTrade STREAM")
    print(f"URL: {url}")
    print("=" * 60)

    async with websockets.connect(
        url,
        compression=None,
        open_timeout=10,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        print("Connected! Waiting for 5 trades...\n")

        for i in range(5):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
            except asyncio.TimeoutError:
                print(f"Timeout after {i} messages")
                break

            print(f"  #{i + 1}: {msg[:300]}")

    print("\naggTrade single stream works!")


async def main() -> None:
    await test_combined()
    await test_single_aggtrade()


if __name__ == "__main__":
    asyncio.run(main())