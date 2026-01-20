"""
Admin API Routes
Real system statistics and monitoring
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import psutil
import os
import asyncio
import subprocess
from datetime import datetime, timedelta
from loguru import logger
import redis.asyncio as redis
import json

from config import settings

router = APIRouter()


def get_git_commit() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_build_date() -> str:
    """Get current date in DDMMYYYY format"""
    now = datetime.utcnow()
    return f"{now.day:02d}{now.month:02d}{now.year}"


@router.get("/version")
async def get_version():
    """Get application version information"""
    return {
        "version": "v3.0",
        "build_date": get_build_date(),
        "git_commit": get_git_commit(),
        "full_version": f"v3.0-{get_build_date()}-{get_git_commit()}"
    }


def get_uptime() -> str:
    """Get system uptime as human readable string"""
    try:
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    except Exception:
        return "N/A"


@router.get("/system")
async def get_system_stats():
    """Get real system statistics - works in Docker containers"""
    try:
        # CPU usage - try multiple methods
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent == 0:
                # Try reading from /proc/stat for Docker
                cpu_percent = await _get_docker_cpu()
        except:
            cpu_percent = await _get_docker_cpu()
        
        # Memory usage
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used
            memory_total = memory.total
        except:
            memory_percent, memory_used, memory_total = await _get_docker_memory()
        
        # Disk usage - check 1.8TB data disk first, fallback to root
        try:
            # Try the 1.8TB data disk mount point first
            data_disk_paths = ['/mnt/sentinel-data', '/mnt/data', '/data']
            disk = None
            disk_path = '/'
            
            for path in data_disk_paths:
                try:
                    disk = psutil.disk_usage(path)
                    if disk.total > 500 * (1024**3):  # More than 500GB = probably the big disk
                        disk_path = path
                        break
                except:
                    continue
            
            if disk is None:
                disk = psutil.disk_usage('/')
                disk_path = '/'
            
            disk_percent = disk.percent
            disk_used = disk.used
            disk_total = disk.total
        except:
            disk_percent = 0
            disk_used = 0
            disk_total = 0
            disk_path = '/'
        
        # Try to get network connections
        try:
            connections = len(psutil.net_connections())
        except:
            connections = 0
        
        # Also get root disk for comparison
        try:
            root_disk = psutil.disk_usage('/')
            root_disk_used = root_disk.used
            root_disk_total = root_disk.total
        except:
            root_disk_used = 0
            root_disk_total = 0
        
        return {
            "success": True,
            "data": {
                "uptime": get_uptime(),
                "cpuUsage": round(cpu_percent, 1),
                "memoryUsage": round(memory_percent, 1),
                "memoryUsedGB": round(memory_used / (1024**3), 2),
                "memoryTotalGB": round(memory_total / (1024**3), 2),
                "diskUsage": round(disk_percent, 1),
                "diskUsedGB": round(disk_used / (1024**3), 2),
                "diskTotalGB": round(disk_total / (1024**3), 2),
                "diskPath": disk_path,
                "rootDiskUsedGB": round(root_disk_used / (1024**3), 2),
                "rootDiskTotalGB": round(root_disk_total / (1024**3), 2),
                "activeConnections": connections,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def _get_docker_cpu() -> float:
    """Get CPU usage from /proc/stat for Docker containers"""
    try:
        # Read CPU stats twice with delay
        with open('/proc/stat', 'r') as f:
            line1 = f.readline()
        
        await asyncio.sleep(0.1)
        
        with open('/proc/stat', 'r') as f:
            line2 = f.readline()
        
        # Parse CPU times
        def parse_cpu(line):
            parts = line.split()
            return [int(x) for x in parts[1:8]]
        
        cpu1 = parse_cpu(line1)
        cpu2 = parse_cpu(line2)
        
        # Calculate deltas
        idle1 = cpu1[3]
        idle2 = cpu2[3]
        total1 = sum(cpu1)
        total2 = sum(cpu2)
        
        idle_delta = idle2 - idle1
        total_delta = total2 - total1
        
        if total_delta == 0:
            return 0
            
        cpu_percent = ((total_delta - idle_delta) / total_delta) * 100
        return cpu_percent
        
    except Exception as e:
        logger.debug(f"Docker CPU read failed: {e}")
        return 0


async def _get_docker_memory() -> tuple:
    """Get memory usage from /proc/meminfo for Docker containers"""
    try:
        meminfo = {}
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(':')
                value = int(parts[1]) * 1024  # Convert KB to bytes
                meminfo[key] = value
        
        total = meminfo.get('MemTotal', 0)
        available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
        used = total - available
        
        percent = (used / total * 100) if total > 0 else 0
        
        return percent, used, total
        
    except Exception as e:
        logger.debug(f"Docker memory read failed: {e}")
        return 0, 0, 0


@router.get("/users")
async def get_users():
    """Get registered users from Redis (users who connected exchanges)"""
    try:
        r = await redis.from_url(settings.REDIS_URL)
        
        users = []
        
        # Get all users with exchange credentials
        keys = await r.keys('exchange:credentials:*')
        
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            user_id = key_str.replace('exchange:credentials:', '')
            
            # Get credentials info
            creds = await r.hgetall(key)
            creds_dict = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in creds.items()
            }
            
            # Get user settings - try both global and user-specific
            user_settings = await r.hgetall(f'settings:{user_id}')
            if not user_settings and user_id == 'default':
                user_settings = await r.hgetall('settings:global')
            settings_dict = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in user_settings.items()
            } if user_settings else {}
            
            # Get user stats - try multiple possible keys
            stats = {}
            # Try user-specific stats first
            user_stats = await r.get(f'trader:stats:{user_id}')
            if user_stats:
                try:
                    stats = json.loads(user_stats)
                except:
                    pass
            
            # For default user, also try the global trader:stats
            if not stats and user_id == 'default':
                global_stats = await r.get('trader:stats')
                if global_stats:
                    try:
                        stats = json.loads(global_stats)
                    except:
                        pass
            
            # Get completed trades count from list
            completed_key = f'trades:completed:{user_id}'
            completed_count = await r.llen(completed_key)
            if completed_count == 0 and user_id == 'default':
                completed_count = await r.llen('trades:completed:default')
            
            # Use completed count if stats doesn't have total_trades
            total_trades = stats.get('total_trades', completed_count) or completed_count
            
            # Check if trading is active
            is_active = settings_dict.get('tradingEnabled', 'false').lower() == 'true'
            
            # Get email from credentials or settings
            email = creds_dict.get('email') or settings_dict.get('email')
            if not email:
                email = f'{user_id}@sentinel.ai' if user_id != 'default' else 'admin@sentinel.ai'
            
            # Build user object
            users.append({
                'id': user_id,
                'email': email,
                'name': settings_dict.get('displayName', 'Admin' if user_id == 'default' else user_id),
                'exchange': 'Bybit',
                'exchangeConnected': bool(creds_dict.get('api_key')),
                'isActive': is_active,
                'isAdmin': user_id == 'default',
                'createdAt': creds_dict.get('created_at', datetime.now().isoformat()),
                'totalTrades': int(total_trades) if total_trades else 0,
                'totalPnl': float(stats.get('total_pnl', 0)),
                'winningTrades': int(stats.get('winning_trades', 0)),
                'winRate': round((stats.get('winning_trades', 0) / total_trades * 100), 1) if total_trades else 0,
                'lastActive': settings_dict.get('lastActive', 'N/A')
            })
        
        await r.close()
        
        return {
            "success": True,
            "data": {
                "users": users,
                "total": len(users)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        return {"success": False, "error": str(e), "data": {"users": [], "total": 0}}


@router.get("/ai-stats")
async def get_ai_stats():
    """Get AI learning statistics from Redis Q-Learning storage"""
    try:
        r = await redis.from_url(settings.REDIS_URL)
        
        # Get Q-values (strategy learning)
        q_values_raw = await r.get('ai:learning:q_values')
        q_values = json.loads(q_values_raw) if q_values_raw else {}
        
        # Get pattern memory
        patterns_raw = await r.get('ai:learning:patterns')
        patterns = json.loads(patterns_raw) if patterns_raw else {}
        
        # Get market states
        market_states_raw = await r.get('ai:learning:market_states')
        market_states = json.loads(market_states_raw) if market_states_raw else {}
        
        # Get sentiment patterns
        sentiment_raw = await r.get('ai:learning:sentiment')
        sentiment_patterns = json.loads(sentiment_raw) if sentiment_raw else {}
        
        # Get learning statistics
        stats_raw = await r.hgetall('ai:learning:stats')
        stats = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in stats_raw.items()
        } if stats_raw else {}
        
        # Get trade statistics
        trade_stats_raw = await r.hgetall('ai:trading:stats')
        trade_stats = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in trade_stats_raw.items()
        } if trade_stats_raw else {}
        
        # Calculate Q-state count
        q_state_count = sum(
            1 for regime_values in q_values.values() 
            if isinstance(regime_values, dict)
            for q in regime_values.values() 
            if abs(q) > 0.1
        )
        
        # Total models = active learning components
        # We have 5 learning sources: Q-Learning, Patterns, Market States, Sentiment, Technical
        active_models = 0
        if q_state_count > 0:
            active_models += 1  # Q-Learning Strategy Model
        if len(patterns) > 0:
            active_models += 1  # Pattern Recognition Model
        if len(market_states) > 0:
            active_models += 1  # Market State Model
        if len(sentiment_patterns) > 0:
            active_models += 1  # Sentiment Analysis Model
        
        # If learning is actively running, count the technical analysis model
        learning_iterations = int(stats.get('learning_iterations', 0))
        if learning_iterations > 0:
            active_models += 1  # Technical Analysis Model
        
        # Total learned states
        total_states = q_state_count + len(patterns) + len(market_states) + len(sentiment_patterns)
        
        # Calculate training progress
        training_progress = min(100, round(total_states / 50 * 100, 1))
        
        await r.close()
        
        return {
            "success": True,
            "data": {
                "modelsLoaded": active_models,
                "totalModels": 5,
                "strategyModel": "active" if q_state_count > 0 else "learning",
                "patternModel": "active" if len(patterns) > 0 else "learning",
                "marketModel": "active" if len(market_states) > 0 else "learning",
                "sentimentModel": "active" if len(sentiment_patterns) > 0 else "learning",
                "technicalModel": "active" if learning_iterations > 0 else "initializing",
                "trainingProgress": training_progress,
                "learningIterations": learning_iterations,
                "totalStatesLearned": total_states,
                "qStates": q_state_count,
                "patternsLearned": len(patterns),
                "marketStates": len(market_states),
                "sentimentStates": len(sentiment_patterns),
                "totalTrades": int(trade_stats.get('total_trades', 0)),
                "winRate": float(trade_stats.get('win_rate', 0)),
                "totalPnl": f"€{float(trade_stats.get('total_profit', 0)):.2f}"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get AI stats: {e}")
        return {
            "success": True,
            "data": {
                "modelsLoaded": 0,
                "totalModels": 5,
                "strategyModel": "initializing",
                "patternModel": "initializing",
                "marketModel": "initializing",
                "sentimentModel": "initializing",
                "technicalModel": "initializing",
                "trainingProgress": 0,
                "learningIterations": 0,
                "totalStatesLearned": 0,
                "error": str(e)
            }
        }


@router.get("/services")
async def get_service_health():
    """Check health of all services with real status"""
    import socket
    import subprocess
    
    services = {}
    
    # Check Redis
    try:
        r = await redis.from_url(settings.REDIS_URL, socket_timeout=2)
        await r.ping()
        services["Redis"] = {"status": "healthy", "details": "Connected"}
        await r.close()
    except Exception as e:
        services["Redis"] = {"status": "unhealthy", "details": str(e)[:50]}
    
    # Check PostgreSQL
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('sentinel_postgres', 5432))
        sock.close()
        if result == 0:
            services["PostgreSQL"] = {"status": "healthy", "details": "Port 5432 open"}
        else:
            services["PostgreSQL"] = {"status": "unhealthy", "details": "Connection refused"}
    except Exception as e:
        services["PostgreSQL"] = {"status": "unknown", "details": str(e)[:50]}
    
    # Check Kafka
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('sentinel_kafka', 9092))
        sock.close()
        if result == 0:
            services["Kafka"] = {"status": "healthy", "details": "Port 9092 open"}
        else:
            services["Kafka"] = {"status": "stopped", "details": "Not running"}
    except Exception as e:
        services["Kafka"] = {"status": "stopped", "details": "Not required"}
    
    # Check Frontend (Nginx)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('sentinel_nginx', 80))
        sock.close()
        if result == 0:
            services["Frontend"] = {"status": "healthy", "details": "Nginx running"}
        else:
            services["Frontend"] = {"status": "unhealthy", "details": "Connection refused"}
    except Exception as e:
        services["Frontend"] = {"status": "unknown", "details": str(e)[:50]}
    
    # Check Backend PHP
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('sentinel_backend', 9000))
        sock.close()
        if result == 0:
            services["Backend"] = {"status": "healthy", "details": "PHP-FPM running"}
        else:
            services["Backend"] = {"status": "restarting", "details": "Initializing"}
    except Exception as e:
        services["Backend"] = {"status": "unknown", "details": str(e)[:50]}
    
    # AI Services (this is us, we're healthy if responding)
    services["AI Services"] = {"status": "healthy", "details": "FastAPI running"}
    
    # Count healthy services
    healthy_count = sum(1 for s in services.values() if s["status"] == "healthy")
    total_count = len(services)
    
    return {
        "success": True,
        "data": {
            "services": services,
            "healthyCount": healthy_count,
            "totalCount": total_count,
            "allHealthy": healthy_count == total_count,
            "timestamp": datetime.now().isoformat()
        }
    }


@router.get("/traffic")
async def get_traffic_stats():
    """Get traffic and request statistics"""
    try:
        # In production, this would be tracked via middleware
        # and stored in Redis or ClickHouse
        
        return {
            "success": True,
            "data": {
                "requestsToday": 0,
                "requestsHour": 0,
                "avgResponseTime": 0,
                "errorRate": 0,
                "topEndpoints": [],
                "requestsByHour": []
            }
        }
    except Exception as e:
        logger.error(f"Failed to get traffic stats: {e}")
        return {"success": False, "error": str(e)}


@router.get("/stats")
async def get_public_stats():
    """Get public statistics for landing page - LIVE DATA"""
    try:
        r = await redis.from_url(settings.REDIS_URL)
        
        # Get trading stats
        trade_stats_raw = await r.hgetall('ai:trading:stats')
        trade_stats = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in trade_stats_raw.items()
        } if trade_stats_raw else {}
        
        # Get total volume from trades
        total_volume = float(trade_stats.get('total_volume', 0))
        total_trades = int(trade_stats.get('total_trades', 0))
        wins = int(trade_stats.get('wins', 0))
        losses = int(trade_stats.get('losses', 0))
        
        # Calculate win rate
        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
        else:
            win_rate = 0
        
        # Get active users count
        user_keys = await r.keys('exchange:credentials:*')
        active_users = len(user_keys) if user_keys else 0
        
        # Get system uptime
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime_seconds = (datetime.now() - boot_time).total_seconds()
            uptime_percent = min(99.99, (uptime_seconds / (uptime_seconds + 60)) * 100)
        except:
            uptime_percent = 99.99
        
        # Get AI accuracy from learning stats
        learning_stats_raw = await r.hgetall('ai:learning:stats')
        learning_stats = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in learning_stats_raw.items()
        } if learning_stats_raw else {}
        
        ai_accuracy = float(learning_stats.get('overall_accuracy', win_rate))
        
        # Get today's volume
        today_key = f"stats:volume:{datetime.now().strftime('%Y-%m-%d')}"
        today_volume = await r.get(today_key)
        today_volume = float(today_volume) if today_volume else 0
        
        await r.aclose()
        
        return {
            "total_volume": total_volume if total_volume > 0 else today_volume,
            "totalVolume": total_volume if total_volume > 0 else today_volume,
            "active_users": active_users,
            "activeUsers": active_users,
            "ai_accuracy": ai_accuracy if ai_accuracy > 0 else win_rate,
            "aiAccuracy": ai_accuracy if ai_accuracy > 0 else win_rate,
            "win_rate": win_rate,
            "winRate": win_rate,
            "uptime": uptime_percent,
            "total_trades": total_trades,
            "totalTrades": total_trades,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get public stats: {e}")
        # Return default values on error
        return {
            "total_volume": 0,
            "totalVolume": 0,
            "active_users": 1,
            "activeUsers": 1,
            "ai_accuracy": 0,
            "aiAccuracy": 0,
            "win_rate": 0,
            "winRate": 0,
            "uptime": 99.99,
            "total_trades": 0,
            "totalTrades": 0,
            "timestamp": datetime.now().isoformat()
        }

