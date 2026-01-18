"""
SENTINEL AI - Cluster Manager
Multi-server load balancing and coordination

This allows you to run multiple servers as ONE unified system:
- Work distribution (symbols split across servers)
- Leader election (one master, rest workers)
- Heartbeat monitoring
- Automatic failover
- Shared state via Redis

Architecture:
- All servers connect to same Redis
- One server is LEADER (coordinates)
- Others are WORKERS (execute)
- Work is distributed by symbol ranges

Adding a new server:
1. Deploy same code
2. Set CLUSTER_NODE_ID environment variable
3. Point to same Redis
4. It auto-joins the cluster
"""

import asyncio
import os
import socket
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import hashlib
from loguru import logger
import redis.asyncio as redis

from config import settings


@dataclass
class ClusterNode:
    """A node in the cluster"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    role: str  # 'leader', 'worker'
    status: str  # 'online', 'offline', 'starting'
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    last_heartbeat: str
    assigned_symbols: List[str]
    current_load: float  # 0-100
    tasks_completed: int
    
    
@dataclass
class ClusterTask:
    """A task distributed across the cluster"""
    task_id: str
    task_type: str  # 'scan', 'analyze', 'train', 'trade'
    symbol: str
    assigned_to: str  # node_id
    priority: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    status: str  # 'pending', 'running', 'completed', 'failed'
    result: Optional[Dict]


class ClusterManager:
    """
    Manages a cluster of SENTINEL nodes
    
    Features:
    - Automatic node discovery
    - Leader election using Redis
    - Work distribution by consistent hashing
    - Health monitoring
    - Automatic failover
    
    Usage:
    1. Set SENTINEL_NODE_ID env var (unique per server)
    2. All nodes connect to same Redis
    3. Cluster auto-organizes
    """
    
    # Redis keys
    NODES_KEY = "cluster:nodes"
    LEADER_KEY = "cluster:leader"
    TASKS_KEY = "cluster:tasks"
    LOCK_KEY = "cluster:lock"
    
    # Timings
    HEARTBEAT_INTERVAL = 5  # seconds
    HEARTBEAT_TIMEOUT = 15  # seconds
    LEADER_TTL = 30  # seconds
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # This node's info
        self.node_id = os.environ.get('SENTINEL_NODE_ID', self._generate_node_id())
        self.hostname = socket.gethostname()
        self.ip_address = self._get_ip_address()
        self.port = int(os.environ.get('PORT', 8000))
        self.role = 'worker'  # Will be updated
        self.cpu_cores = os.cpu_count() or 4
        self.memory_gb = self._get_memory_gb()
        self.gpu_available = self._check_gpu()
        
        # Cluster state
        self.nodes: Dict[str, ClusterNode] = {}
        self.is_leader = False
        self.leader_id: Optional[str] = None
        
        # Assigned work
        self.assigned_symbols: List[str] = []
        self.current_load = 0.0
        self.tasks_completed = 0
        
        # Callbacks
        self.on_symbols_assigned = None
        self.on_become_leader = None
        self.on_become_worker = None
        
    async def initialize(self):
        """Initialize cluster manager"""
        logger.info(f"Initializing Cluster Manager (Node: {self.node_id})...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        self.is_running = True
        
        # Register this node
        await self._register_node()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._leader_election_loop())
        asyncio.create_task(self._work_distribution_loop())
        asyncio.create_task(self._cluster_monitor_loop())
        
        logger.info(f"Cluster Manager initialized - Node {self.node_id} online")
        
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"Shutting down Cluster Manager (Node: {self.node_id})...")
        self.is_running = False
        
        # Unregister this node
        await self._unregister_node()
        
        # If leader, release leadership
        if self.is_leader:
            await self._release_leadership()
            
        if self.redis_client:
            await self.redis_client.aclose()
            
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        hostname = socket.gethostname()
        pid = os.getpid()
        return hashlib.md5(f"{hostname}-{pid}".encode()).hexdigest()[:8]
        
    def _get_ip_address(self) -> str:
        """Get this node's IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
            
    def _get_memory_gb(self) -> float:
        """Get available memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 8.0
            
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
            
    async def _register_node(self):
        """Register this node in the cluster"""
        node = ClusterNode(
            node_id=self.node_id,
            hostname=self.hostname,
            ip_address=self.ip_address,
            port=self.port,
            role='worker',
            status='starting',
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            gpu_available=self.gpu_available,
            last_heartbeat=datetime.utcnow().isoformat(),
            assigned_symbols=[],
            current_load=0.0,
            tasks_completed=0
        )
        
        await self.redis_client.hset(
            self.NODES_KEY,
            self.node_id,
            json.dumps(asdict(node))
        )
        
        self.nodes[self.node_id] = node
        logger.info(f"Node {self.node_id} registered in cluster")
        
    async def _unregister_node(self):
        """Unregister this node from cluster"""
        await self.redis_client.hdel(self.NODES_KEY, self.node_id)
        logger.info(f"Node {self.node_id} unregistered from cluster")
        
    async def _heartbeat_loop(self):
        """Send heartbeat to indicate this node is alive"""
        while self.is_running:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1)
                
    async def _send_heartbeat(self):
        """Send heartbeat update"""
        try:
            # Update this node's info
            node_data = await self.redis_client.hget(self.NODES_KEY, self.node_id)
            if node_data:
                node = ClusterNode(**json.loads(node_data))
                node.last_heartbeat = datetime.utcnow().isoformat()
                node.status = 'online'
                node.role = 'leader' if self.is_leader else 'worker'
                node.current_load = self.current_load
                node.tasks_completed = self.tasks_completed
                node.assigned_symbols = self.assigned_symbols
                
                await self.redis_client.hset(
                    self.NODES_KEY,
                    self.node_id,
                    json.dumps(asdict(node))
                )
        except Exception as e:
            logger.debug(f"Heartbeat send error: {e}")
            
    async def _leader_election_loop(self):
        """Try to become or maintain leader"""
        while self.is_running:
            try:
                await self._attempt_leadership()
                await asyncio.sleep(self.LEADER_TTL / 3)
            except Exception as e:
                logger.error(f"Leader election error: {e}")
                await asyncio.sleep(5)
                
    async def _attempt_leadership(self):
        """Attempt to become the leader"""
        try:
            # Try to set leadership key with NX (only if not exists)
            acquired = await self.redis_client.set(
                self.LEADER_KEY,
                self.node_id,
                nx=True,
                ex=self.LEADER_TTL
            )
            
            if acquired:
                if not self.is_leader:
                    self.is_leader = True
                    self.leader_id = self.node_id
                    self.role = 'leader'
                    logger.info(f"Node {self.node_id} elected as LEADER")
                    
                    if self.on_become_leader:
                        await self.on_become_leader()
            else:
                # Check if we're still the leader
                current_leader = await self.redis_client.get(self.LEADER_KEY)
                if current_leader:
                    current_leader = current_leader.decode()
                    self.leader_id = current_leader
                    
                    if current_leader == self.node_id:
                        # Refresh our leadership
                        await self.redis_client.expire(self.LEADER_KEY, self.LEADER_TTL)
                    else:
                        if self.is_leader:
                            self.is_leader = False
                            self.role = 'worker'
                            logger.info(f"Node {self.node_id} demoted to WORKER")
                            
                            if self.on_become_worker:
                                await self.on_become_worker()
                                
        except Exception as e:
            logger.debug(f"Leadership attempt error: {e}")
            
    async def _release_leadership(self):
        """Release leadership"""
        try:
            current = await self.redis_client.get(self.LEADER_KEY)
            if current and current.decode() == self.node_id:
                await self.redis_client.delete(self.LEADER_KEY)
                logger.info(f"Node {self.node_id} released leadership")
        except:
            pass
            
    async def _work_distribution_loop(self):
        """Distribute work across nodes (leader only)"""
        while self.is_running:
            try:
                if self.is_leader:
                    await self._distribute_work()
                await asyncio.sleep(30)  # Redistribute every 30s
            except Exception as e:
                logger.error(f"Work distribution error: {e}")
                await asyncio.sleep(10)
                
    async def _distribute_work(self):
        """Distribute symbols across online nodes"""
        if not self.is_leader:
            return
            
        try:
            # Get all online nodes
            online_nodes = await self._get_online_nodes()
            if not online_nodes:
                return
                
            # Get all symbols to distribute
            symbols_data = await self.redis_client.get('trading:available_symbols')
            if not symbols_data:
                return
                
            symbols = symbols_data.decode().split(',')
            
            # Calculate weight for each node based on CPU cores
            total_weight = sum(n.cpu_cores for n in online_nodes)
            
            # Distribute symbols proportionally
            symbol_index = 0
            for node in online_nodes:
                # Calculate how many symbols this node should handle
                weight_ratio = node.cpu_cores / total_weight
                num_symbols = int(len(symbols) * weight_ratio)
                
                # Ensure at least some symbols for each node
                num_symbols = max(num_symbols, 10)
                
                # Assign symbols
                end_index = min(symbol_index + num_symbols, len(symbols))
                assigned = symbols[symbol_index:end_index]
                
                # Store assignment in Redis
                await self.redis_client.set(
                    f"cluster:assignments:{node.node_id}",
                    ','.join(assigned),
                    ex=60  # 1 minute TTL
                )
                
                symbol_index = end_index
                
                if symbol_index >= len(symbols):
                    break
                    
            logger.debug(f"Work distributed to {len(online_nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Work distribution error: {e}")
            
    async def _cluster_monitor_loop(self):
        """Monitor cluster health and load assignments"""
        while self.is_running:
            try:
                await self._refresh_cluster_state()
                await self._load_my_assignments()
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Cluster monitor error: {e}")
                await asyncio.sleep(5)
                
    async def _refresh_cluster_state(self):
        """Refresh view of all cluster nodes"""
        try:
            all_nodes = await self.redis_client.hgetall(self.NODES_KEY)
            
            now = datetime.utcnow()
            online_count = 0
            
            for node_id, node_data in all_nodes.items():
                node_id = node_id.decode() if isinstance(node_id, bytes) else node_id
                node = ClusterNode(**json.loads(node_data))
                
                # Check if node is stale
                last_hb = datetime.fromisoformat(node.last_heartbeat)
                if (now - last_hb).total_seconds() > self.HEARTBEAT_TIMEOUT:
                    node.status = 'offline'
                else:
                    online_count += 1
                    
                self.nodes[node_id] = node
                
            # Clean up offline nodes (leader only)
            if self.is_leader:
                for node_id, node in list(self.nodes.items()):
                    if node.status == 'offline':
                        last_hb = datetime.fromisoformat(node.last_heartbeat)
                        if (now - last_hb).total_seconds() > self.HEARTBEAT_TIMEOUT * 3:
                            await self.redis_client.hdel(self.NODES_KEY, node_id)
                            del self.nodes[node_id]
                            logger.info(f"Removed stale node: {node_id}")
                            
        except Exception as e:
            logger.debug(f"Cluster state refresh error: {e}")
            
    async def _load_my_assignments(self):
        """Load this node's symbol assignments"""
        try:
            assignments = await self.redis_client.get(f"cluster:assignments:{self.node_id}")
            if assignments:
                new_symbols = assignments.decode().split(',')
                
                if new_symbols != self.assigned_symbols:
                    self.assigned_symbols = new_symbols
                    logger.info(f"Node {self.node_id} assigned {len(new_symbols)} symbols")
                    
                    if self.on_symbols_assigned:
                        await self.on_symbols_assigned(new_symbols)
                        
        except Exception as e:
            logger.debug(f"Load assignments error: {e}")
            
    async def _get_online_nodes(self) -> List[ClusterNode]:
        """Get list of online nodes"""
        return [n for n in self.nodes.values() if n.status == 'online']
        
    async def submit_task(self, task_type: str, symbol: str, priority: int = 1) -> str:
        """Submit a task to the cluster"""
        task_id = f"{task_type}:{symbol}:{datetime.utcnow().timestamp()}"
        
        # Find best node for this task
        target_node = await self._find_best_node_for_symbol(symbol)
        
        task = ClusterTask(
            task_id=task_id,
            task_type=task_type,
            symbol=symbol,
            assigned_to=target_node,
            priority=priority,
            created_at=datetime.utcnow().isoformat(),
            started_at=None,
            completed_at=None,
            status='pending',
            result=None
        )
        
        await self.redis_client.lpush(
            f"cluster:tasks:{target_node}",
            json.dumps(asdict(task))
        )
        
        return task_id
        
    async def _find_best_node_for_symbol(self, symbol: str) -> str:
        """Find the best node to handle a symbol"""
        # Check if symbol is assigned to a specific node
        for node_id, node in self.nodes.items():
            if node.status == 'online' and symbol in node.assigned_symbols:
                return node_id
                
        # Fallback to least loaded online node
        online = await self._get_online_nodes()
        if online:
            return min(online, key=lambda n: n.current_load).node_id
            
        return self.node_id  # Fallback to self
        
    async def get_my_tasks(self) -> List[ClusterTask]:
        """Get tasks assigned to this node"""
        tasks = []
        try:
            task_data = await self.redis_client.lrange(f"cluster:tasks:{self.node_id}", 0, 99)
            for data in task_data:
                tasks.append(ClusterTask(**json.loads(data)))
        except:
            pass
        return tasks
        
    async def complete_task(self, task_id: str, result: Dict):
        """Mark a task as completed"""
        try:
            # Remove from queue (simplified - in production use better queue)
            self.tasks_completed += 1
            
            # Store result
            await self.redis_client.set(
                f"cluster:result:{task_id}",
                json.dumps(result),
                ex=3600  # 1 hour
            )
        except:
            pass
            
    async def get_cluster_status(self) -> Dict:
        """Get cluster status"""
        online_nodes = await self._get_online_nodes()
        
        return {
            'this_node': {
                'node_id': self.node_id,
                'role': 'leader' if self.is_leader else 'worker',
                'hostname': self.hostname,
                'ip_address': self.ip_address,
                'assigned_symbols': len(self.assigned_symbols),
                'current_load': self.current_load,
                'tasks_completed': self.tasks_completed
            },
            'cluster': {
                'total_nodes': len(self.nodes),
                'online_nodes': len(online_nodes),
                'leader_id': self.leader_id,
                'total_cpu_cores': sum(n.cpu_cores for n in online_nodes),
                'total_memory_gb': sum(n.memory_gb for n in online_nodes),
                'gpu_nodes': sum(1 for n in online_nodes if n.gpu_available)
            },
            'nodes': [
                {
                    'node_id': n.node_id,
                    'hostname': n.hostname,
                    'role': n.role,
                    'status': n.status,
                    'cpu_cores': n.cpu_cores,
                    'memory_gb': n.memory_gb,
                    'gpu': n.gpu_available,
                    'load': n.current_load,
                    'symbols': len(n.assigned_symbols)
                }
                for n in self.nodes.values()
            ]
        }


# Global instance
cluster_manager = ClusterManager()

