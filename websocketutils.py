from typing import List
from fastapi import WebSocket


# websocket管理器
class WebsocketManager:
    # 初始化
    def __init__(self):
        self.active_clients: List[WebSocket] = []

    # 连接时将websocket连接对象加入管理器
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_clients.append(websocket)

    # 断连时将websocket连接对象从管理器删除
    def disconnect(self, websocket: WebSocket):
        self.active_clients.remove(websocket)

    # 单独发送消息给客户端
    async def send_message_to_client(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    # 消息广播
    async def broadcast(self, message: str):
        for connection in self.active_clients:
            await connection.send_text(message)

