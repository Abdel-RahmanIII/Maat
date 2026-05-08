import asyncio
import threading

import pytest
from fastapi import WebSocket

from src.runner.api.ws import ConnectionManager


class FakeWebSocket:
    def __init__(self):
        self.accepted = False
        self.messages = []
        
    async def accept(self):
        self.accepted = True
        
    async def send_json(self, data):
        self.messages.append(data)


def test_connection_manager_connect_disconnect():
    cm = ConnectionManager()
    ws = FakeWebSocket()
    
    async def run():
        await cm.connect(ws)
        assert ws.accepted
        assert ws in cm._connections
        
        cm.disconnect(ws)
        assert ws not in cm._connections
        
    asyncio.run(run())


def test_connection_manager_broadcast_sync():
    cm = ConnectionManager()
    
    # We need an event loop running in a thread so broadcast_sync can schedule onto it
    loop = asyncio.new_event_loop()
    
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()
        
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    
    cm.set_event_loop(loop)
    
    ws1 = FakeWebSocket()
    ws2 = FakeWebSocket()
    
    cm._connections.append(ws1)
    cm._connections.append(ws2)
    
    cm.broadcast_sync({"hello": "world"})
    
    # Give the event loop a little time to process the threadsafe calls
    # Wait synchronously by busy waiting
    import time
    for _ in range(10):
        if ws1.messages and ws2.messages:
            break
        time.sleep(0.01)
    
    assert {"hello": "world"} in ws1.messages
    assert {"hello": "world"} in ws2.messages
    
    loop.call_soon_threadsafe(loop.stop)
    t.join()
