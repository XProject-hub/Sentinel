"""
SENTINEL AI - WebSocket Connection Manager
Real-time communication with clients
"""

from typing import Dict, List, Any, Optional
from fastapi import WebSocket
import asyncio
import json
from loguru import logger
from datetime import datetime


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates:
    - Dashboard data streaming
    - Trade notifications
    - Risk alerts
    - AI status updates
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_subscriptions: Dict[str, List[str]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_subscriptions[user_id] = ['dashboard', 'trades', 'alerts']
        logger.info(f"WebSocket connected: {user_id}")
        
    def disconnect(self, user_id: str):
        """Remove a WebSocket connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_subscriptions:
            del self.user_subscriptions[user_id]
        logger.info(f"WebSocket disconnected: {user_id}")
        
    async def send(self, user_id: str, data: Dict[str, Any]):
        """Send data to a specific user"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(data)
            except Exception as e:
                logger.error(f"WebSocket send error for {user_id}: {e}")
                self.disconnect(user_id)
                
    async def broadcast(self, data: Dict[str, Any], channel: str = 'all'):
        """Broadcast data to all connected users subscribed to channel"""
        disconnected = []
        
        for user_id, websocket in self.active_connections.items():
            # Check if user is subscribed to this channel
            if channel == 'all' or channel in self.user_subscriptions.get(user_id, []):
                try:
                    await websocket.send_json(data)
                except Exception as e:
                    logger.error(f"Broadcast error for {user_id}: {e}")
                    disconnected.append(user_id)
                    
        # Cleanup disconnected users
        for user_id in disconnected:
            self.disconnect(user_id)
            
    async def receive(self, websocket: WebSocket) -> Dict[str, Any]:
        """Receive data from WebSocket"""
        data = await websocket.receive_json()
        return data
        
    async def send_dashboard_update(self, user_id: str, dashboard_data: Dict[str, Any]):
        """Send dashboard update to user"""
        await self.send(user_id, {
            'type': 'dashboard_update',
            'data': dashboard_data,
            'timestamp': datetime.utcnow().isoformat(),
        })
        
    async def send_trade_notification(
        self, 
        user_id: str, 
        trade_type: str,
        trade_data: Dict[str, Any]
    ):
        """Send trade notification"""
        await self.send(user_id, {
            'type': 'trade_notification',
            'trade_type': trade_type,  # opened, closed, updated
            'data': trade_data,
            'timestamp': datetime.utcnow().isoformat(),
        })
        
    async def send_risk_alert(
        self,
        user_id: str,
        alert_type: str,
        severity: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """Send risk alert to user"""
        await self.send(user_id, {
            'type': 'risk_alert',
            'alert_type': alert_type,
            'severity': severity,  # info, warning, critical
            'message': message,
            'data': data or {},
            'timestamp': datetime.utcnow().isoformat(),
        })
        
    async def send_ai_insight(
        self,
        user_id: str,
        insight: str,
        confidence: float,
        action: Optional[str] = None
    ):
        """Send AI insight update"""
        await self.send(user_id, {
            'type': 'ai_insight',
            'insight': insight,
            'confidence': confidence,
            'action': action,
            'timestamp': datetime.utcnow().isoformat(),
        })
        
    async def send_price_update(self, symbol: str, price_data: Dict[str, Any]):
        """Broadcast price update to all users"""
        await self.broadcast({
            'type': 'price_update',
            'symbol': symbol,
            'data': price_data,
            'timestamp': datetime.utcnow().isoformat(),
        }, channel='prices')
        
    def subscribe(self, user_id: str, channels: List[str]):
        """Subscribe user to specific channels"""
        if user_id in self.user_subscriptions:
            self.user_subscriptions[user_id] = list(set(
                self.user_subscriptions[user_id] + channels
            ))
            
    def unsubscribe(self, user_id: str, channels: List[str]):
        """Unsubscribe user from channels"""
        if user_id in self.user_subscriptions:
            self.user_subscriptions[user_id] = [
                c for c in self.user_subscriptions[user_id]
                if c not in channels
            ]
            
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
        
    def is_connected(self, user_id: str) -> bool:
        """Check if user is connected"""
        return user_id in self.active_connections

