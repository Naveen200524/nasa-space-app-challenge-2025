"""
WebSocket real-time streaming for live seismic monitoring.
Based on api.md specifications for real-time data streams and event notifications.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
import websockets
import websockets.exceptions
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of real-time streams."""
    SEISMIC_DATA = "seismic_data"
    EARTHQUAKE_ALERTS = "earthquake_alerts"
    DETECTION_EVENTS = "detection_events"
    SYSTEM_STATUS = "system_status"


@dataclass
class StreamMessage:
    """Structure for WebSocket stream messages."""
    stream_type: str
    timestamp: str
    data: Dict[str, Any]
    source: str = "seismoguard"
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))


class WebSocketStreamer:
    """
    WebSocket streaming server for real-time seismic data and alerts.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.subscriptions: Dict[websockets.WebSocketServerProtocol, Set[StreamType]] = {}
        self.server = None
        self.running = False
        
        # Data buffers for different streams
        self.earthquake_buffer: List[Dict] = []
        self.detection_buffer: List[Dict] = []
        self.system_status = {
            'status': 'running',
            'connected_clients': 0,
            'uptime': 0,
            'last_update': datetime.utcnow().isoformat()
        }
        
        # Background tasks
        self.tasks: List[asyncio.Task] = []
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        self.subscriptions[websocket] = set()
        
        logger.info(f"Client connected from {websocket.remote_address}")
        self.system_status['connected_clients'] = len(self.clients)
        
        try:
            # Send welcome message
            welcome_msg = StreamMessage(
                stream_type="system",
                timestamp=datetime.utcnow().isoformat(),
                data={
                    'message': 'Connected to SeismoGuard WebSocket stream',
                    'available_streams': [stream.value for stream in StreamType],
                    'client_id': id(websocket)
                }
            )
            await websocket.send(welcome_msg.to_json())
            
            # Handle client messages
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {websocket.remote_address} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {websocket.remote_address}: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a WebSocket client."""
        self.clients.discard(websocket)
        self.subscriptions.pop(websocket, None)
        self.system_status['connected_clients'] = len(self.clients)
        logger.info(f"Client {websocket.remote_address} unregistered")
    
    async def handle_client_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming messages from clients."""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'subscribe':
                streams = data.get('streams', [])
                for stream_name in streams:
                    try:
                        stream_type = StreamType(stream_name)
                        self.subscriptions[websocket].add(stream_type)
                        logger.info(f"Client subscribed to {stream_name}")
                    except ValueError:
                        logger.warning(f"Unknown stream type: {stream_name}")
                
                # Send confirmation
                response = StreamMessage(
                    stream_type="system",
                    timestamp=datetime.utcnow().isoformat(),
                    data={
                        'message': 'Subscription updated',
                        'subscribed_streams': list(self.subscriptions[websocket])
                    }
                )
                await websocket.send(response.to_json())
            
            elif action == 'unsubscribe':
                streams = data.get('streams', [])
                for stream_name in streams:
                    try:
                        stream_type = StreamType(stream_name)
                        self.subscriptions[websocket].discard(stream_type)
                        logger.info(f"Client unsubscribed from {stream_name}")
                    except ValueError:
                        logger.warning(f"Unknown stream type: {stream_name}")
            
            elif action == 'ping':
                # Respond to ping
                pong = StreamMessage(
                    stream_type="system",
                    timestamp=datetime.utcnow().isoformat(),
                    data={'message': 'pong'}
                )
                await websocket.send(pong.to_json())
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling message from {websocket.remote_address}: {e}")
    
    async def broadcast_to_subscribers(self, stream_type: StreamType, data: Dict[str, Any]):
        """Broadcast data to all subscribers of a stream type."""
        if not self.clients:
            return
        
        message = StreamMessage(
            stream_type=stream_type.value,
            timestamp=datetime.utcnow().isoformat(),
            data=data
        )
        
        # Find subscribers
        subscribers = [
            client for client, subscriptions in self.subscriptions.items()
            if stream_type in subscriptions
        ]
        
        if subscribers:
            # Send to all subscribers
            await asyncio.gather(
                *[self.send_safe(client, message.to_json()) for client in subscribers],
                return_exceptions=True
            )
            logger.debug(f"Broadcasted {stream_type.value} to {len(subscribers)} clients")
    
    async def send_safe(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Send message to client with error handling."""
        try:
            await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
    
    async def earthquake_alert_stream(self):
        """Background task for earthquake alerts."""
        from .data_integration_hub import DataIntegrationHub, DataSource
        
        data_hub = DataIntegrationHub()
        last_check = datetime.utcnow()
        
        while self.running:
            try:
                # Check for new earthquakes every 60 seconds
                await asyncio.sleep(60)
                
                # Get recent earthquakes
                earthquakes = await data_hub.get_recent_earthquakes(
                    magnitude=4.0, 
                    hours=1,
                    sources=[DataSource.USGS_REALTIME, DataSource.EMSC]
                )
                
                # Filter for new earthquakes since last check
                new_earthquakes = [
                    eq for eq in earthquakes 
                    if eq['time'] > last_check
                ]
                
                if new_earthquakes:
                    for earthquake in new_earthquakes:
                        # Convert datetime to string for JSON serialization
                        eq_data = earthquake.copy()
                        if hasattr(eq_data['time'], 'isoformat'):
                            eq_data['time'] = eq_data['time'].isoformat()
                        
                        await self.broadcast_to_subscribers(
                            StreamType.EARTHQUAKE_ALERTS,
                            {
                                'alert_type': 'new_earthquake',
                                'earthquake': eq_data,
                                'severity': 'high' if earthquake['magnitude'] > 6.0 else 'medium'
                            }
                        )
                
                last_check = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Error in earthquake alert stream: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def system_status_stream(self):
        """Background task for system status updates."""
        start_time = time.time()
        
        while self.running:
            try:
                # Update system status every 30 seconds
                await asyncio.sleep(30)
                
                self.system_status.update({
                    'uptime': int(time.time() - start_time),
                    'connected_clients': len(self.clients),
                    'last_update': datetime.utcnow().isoformat(),
                    'memory_usage': self._get_memory_usage()
                })
                
                await self.broadcast_to_subscribers(
                    StreamType.SYSTEM_STATUS,
                    self.system_status
                )
                
            except Exception as e:
                logger.error(f"Error in system status stream: {e}")
                await asyncio.sleep(30)
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self.register_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        self.running = True
        
        # Start background tasks
        self.tasks = [
            asyncio.create_task(self.earthquake_alert_stream()),
            asyncio.create_task(self.system_status_stream())
        ]
        
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        logger.info("Stopping WebSocket server...")
        
        self.running = False
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        
        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    def broadcast_detection_event(self, event_data: Dict[str, Any]):
        """Broadcast a detection event to subscribers."""
        if self.running:
            asyncio.create_task(
                self.broadcast_to_subscribers(
                    StreamType.DETECTION_EVENTS,
                    event_data
                )
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current server status."""
        return {
            'running': self.running,
            'host': self.host,
            'port': self.port,
            'connected_clients': len(self.clients),
            'system_status': self.system_status
        }


# Global WebSocket streamer instance
websocket_streamer = WebSocketStreamer()


def start_websocket_server_thread():
    """Start WebSocket server in a separate thread."""
    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(websocket_streamer.start_server())
            loop.run_forever()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info("WebSocket server thread started")
    return thread
