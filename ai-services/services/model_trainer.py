"""
SENTINEL AI - Model Trainer Service
Periodic training coordinator - NOT non-stop training

This is the CORRECT approach:
- Analysis runs 24/7
- Trading runs 24/7  
- Training runs PERIODICALLY (scheduled)

Training Schedule (CPU-optimized):
- XGBoost Edge Classifier: every 6-12 hours
- Regime Model: once daily
- RL Agent: once daily (offline)
- Sentiment: rarely (uses pre-trained)

Resource Allocation:
- 8 cores: live trading + inference
- 8 cores: scanning + feature generation
- 6 cores: background training
- 2 cores: OS / logging / safety
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json
import os
from loguru import logger
import redis.asyncio as redis

from config import settings


@dataclass
class TrainingJob:
    """A scheduled training job"""
    job_id: str
    model_name: str
    interval_hours: int
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    status: str  # 'pending', 'running', 'completed', 'failed'
    priority: int  # 1=high, 2=medium, 3=low
    cpu_cores: int  # How many cores to use
    estimated_duration_minutes: int
    
    
@dataclass
class TrainingMetrics:
    """Metrics from a training run"""
    job_id: str
    model_name: str
    started_at: str
    completed_at: str
    duration_seconds: int
    samples_used: int
    accuracy_before: float
    accuracy_after: float
    improvement: float
    status: str
    error: Optional[str]


class ModelTrainer:
    """
    Coordinates periodic training of all ML models
    
    Ensures:
    - Training never blocks live trading
    - CPU resources are properly allocated
    - Models are trained on validated data
    - Training is logged and monitored
    """
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # Training jobs
        self.jobs: Dict[str, TrainingJob] = {}
        
        # Currently running training
        self.active_training: Optional[str] = None
        self.training_lock = asyncio.Lock()
        
        # Resource limits
        self.max_training_cores = 6
        self.training_hours_start = 0  # UTC hour to allow heavy training (0 = midnight)
        self.training_hours_end = 6    # UTC hour to stop heavy training
        
        # Paths
        self.models_dir = Path("/opt/sentinel/models")
        self.checkpoints_dir = self.models_dir / "checkpoints"
        
        # Stats
        self.training_history: List[TrainingMetrics] = []
        
    async def initialize(self):
        """Initialize trainer"""
        logger.info("Initializing Model Trainer (Periodic Training Coordinator)...")
        
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Define training jobs
        self._define_training_jobs()
        
        # Load job states from Redis
        await self._load_job_states()
        
        self.is_running = True
        
        # Start scheduler
        asyncio.create_task(self._scheduler_loop())
        
        logger.info(f"Model Trainer initialized - {len(self.jobs)} jobs scheduled")
        
    async def shutdown(self):
        """Cleanup"""
        self.is_running = False
        await self._save_job_states()
        if self.redis_client:
            await self.redis_client.aclose()
            
    def _define_training_jobs(self):
        """Define all training jobs with schedules"""
        
        # XGBoost Edge Classifier - Most frequent
        self.jobs['xgboost_edge'] = TrainingJob(
            job_id='xgboost_edge',
            model_name='XGBoost Edge Classifier',
            interval_hours=12,
            last_run=None,
            next_run=None,
            status='pending',
            priority=1,
            cpu_cores=4,
            estimated_duration_minutes=15
        )
        
        # Regime Detection Model
        self.jobs['regime_model'] = TrainingJob(
            job_id='regime_model',
            model_name='Regime Detection Model',
            interval_hours=24,
            last_run=None,
            next_run=None,
            status='pending',
            priority=2,
            cpu_cores=4,
            estimated_duration_minutes=30
        )
        
        # Learning Engine (Q-Learning)
        self.jobs['learning_engine'] = TrainingJob(
            job_id='learning_engine',
            model_name='Q-Learning Engine',
            interval_hours=6,
            last_run=None,
            next_run=None,
            status='pending',
            priority=1,
            cpu_cores=2,
            estimated_duration_minutes=10
        )
        
        # LSTM Price Predictor
        self.jobs['lstm_predictor'] = TrainingJob(
            job_id='lstm_predictor',
            model_name='LSTM Price Predictor',
            interval_hours=24,
            last_run=None,
            next_run=None,
            status='pending',
            priority=2,
            cpu_cores=6,
            estimated_duration_minutes=60
        )
        
        # Pattern Recognition (rule-based, no training needed)
        # FinBERT (pre-trained, no training needed)
        
        logger.info(f"Defined {len(self.jobs)} training jobs")
        
    async def _scheduler_loop(self):
        """Main scheduler loop - runs continuously"""
        logger.info("Starting Model Trainer scheduler...")
        
        while self.is_running:
            try:
                # Check each job
                for job_id, job in self.jobs.items():
                    if self._should_run_job(job):
                        await self._queue_job(job)
                        
                # Process queue (one at a time)
                await self._process_queue()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(300)
                
    def _should_run_job(self, job: TrainingJob) -> bool:
        """Check if job should be run"""
        if job.status == 'running':
            return False
            
        now = datetime.utcnow()
        
        # Never run before
        if job.last_run is None:
            return True
            
        # Check interval
        elapsed = (now - job.last_run).total_seconds() / 3600
        if elapsed < job.interval_hours:
            return False
            
        # For heavy training, prefer off-peak hours
        if job.estimated_duration_minutes > 30:
            current_hour = now.hour
            if not (self.training_hours_start <= current_hour < self.training_hours_end):
                # Allow if very overdue (2x interval)
                if elapsed < job.interval_hours * 2:
                    return False
                    
        return True
        
    async def _queue_job(self, job: TrainingJob):
        """Queue a job for execution"""
        job.status = 'pending'
        job.next_run = datetime.utcnow()
        
        await self.redis_client.lpush('trainer:queue', job.job_id)
        logger.info(f"Queued training job: {job.model_name}")
        
    async def _process_queue(self):
        """Process queued training jobs (one at a time)"""
        if self.active_training:
            return  # Already training
            
        try:
            # Get next job from queue
            job_id = await self.redis_client.rpop('trainer:queue')
            if not job_id:
                return
                
            job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
            
            if job_id not in self.jobs:
                return
                
            job = self.jobs[job_id]
            
            # Run training in background
            asyncio.create_task(self._run_training_job(job))
            
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
            
    async def _run_training_job(self, job: TrainingJob):
        """Run a training job"""
        async with self.training_lock:
            self.active_training = job.job_id
            job.status = 'running'
            
            started_at = datetime.utcnow()
            accuracy_before = await self._get_model_accuracy(job.job_id)
            samples_used = 0
            error = None
            
            logger.info(f"Starting training: {job.model_name} (est. {job.estimated_duration_minutes} min)")
            
            try:
                # Set CPU affinity for training
                os.environ['OMP_NUM_THREADS'] = str(job.cpu_cores)
                
                # Run appropriate training
                if job.job_id == 'xgboost_edge':
                    samples_used = await self._train_xgboost()
                elif job.job_id == 'regime_model':
                    samples_used = await self._train_regime_model()
                elif job.job_id == 'learning_engine':
                    samples_used = await self._train_learning_engine()
                elif job.job_id == 'lstm_predictor':
                    samples_used = await self._train_lstm()
                else:
                    logger.warning(f"Unknown training job: {job.job_id}")
                    
                job.status = 'completed'
                job.last_run = datetime.utcnow()
                
            except Exception as e:
                error = str(e)
                job.status = 'failed'
                logger.error(f"Training failed for {job.model_name}: {e}")
                
            finally:
                self.active_training = None
                
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).seconds
            accuracy_after = await self._get_model_accuracy(job.job_id)
            
            # Record metrics
            metrics = TrainingMetrics(
                job_id=job.job_id,
                model_name=job.model_name,
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                duration_seconds=duration,
                samples_used=samples_used,
                accuracy_before=accuracy_before,
                accuracy_after=accuracy_after,
                improvement=accuracy_after - accuracy_before,
                status=job.status,
                error=error
            )
            
            self.training_history.append(metrics)
            
            # Store in Redis
            await self.redis_client.lpush(
                'trainer:history',
                json.dumps({
                    'job_id': metrics.job_id,
                    'model_name': metrics.model_name,
                    'started_at': metrics.started_at,
                    'completed_at': metrics.completed_at,
                    'duration_seconds': metrics.duration_seconds,
                    'samples_used': metrics.samples_used,
                    'accuracy_before': metrics.accuracy_before,
                    'accuracy_after': metrics.accuracy_after,
                    'improvement': metrics.improvement,
                    'status': metrics.status,
                    'error': metrics.error
                })
            )
            await self.redis_client.ltrim('trainer:history', 0, 99)
            
            logger.info(f"Training completed: {job.model_name} in {duration}s "
                       f"(accuracy: {accuracy_before:.2%} -> {accuracy_after:.2%})")
                       
    async def _train_xgboost(self) -> int:
        """Train XGBoost Edge Classifier"""
        try:
            from services.xgboost_classifier import xgboost_classifier
            
            if xgboost_classifier:
                await xgboost_classifier._train_model()
                return xgboost_classifier.training_samples
        except Exception as e:
            logger.error(f"XGBoost training error: {e}")
        return 0
        
    async def _train_regime_model(self) -> int:
        """Train Regime Detection Model"""
        try:
            # Import regime detector and trigger training
            from services.regime_detector import regime_detector
            
            # Get all active symbols
            symbols_data = await self.redis_client.get('trading:available_symbols')
            symbols = symbols_data.decode().split(',')[:30] if symbols_data else ['BTCUSDT', 'ETHUSDT']
            
            samples = 0
            for symbol in symbols:
                for tf_name in ['1h', '4h']:
                    try:
                        await regime_detector._train_or_update_model(symbol, tf_name)
                        samples += 200  # Approximate samples per model
                    except:
                        pass
                        
            return samples
            
        except Exception as e:
            logger.error(f"Regime model training error: {e}")
        return 0
        
    async def _train_learning_engine(self) -> int:
        """Train Q-Learning Engine"""
        try:
            from services.learning_engine import learning_engine
            
            if learning_engine:
                # Get recent trades for learning
                trades_data = await self.redis_client.lrange('trades:completed', 0, 999)
                
                for trade_json in trades_data:
                    trade = json.loads(trade_json)
                    
                    # Create state from trade data
                    state = {
                        'regime': trade.get('regime', 'unknown'),
                        'trend_strength': trade.get('trend_strength', 0),
                        'volatility': trade.get('volatility', 0),
                        'rsi': trade.get('rsi', 50),
                        'sentiment': trade.get('sentiment', 0)
                    }
                    
                    # Calculate reward from outcome
                    pnl = trade.get('pnl_percent', 0)
                    reward = pnl  # Simple: reward = PnL
                    
                    action = trade.get('action', 'hold')
                    
                    # Update Q-values
                    await learning_engine.update(state, action, reward, {})
                    
                return len(trades_data)
                
        except Exception as e:
            logger.error(f"Learning engine training error: {e}")
        return 0
        
    async def _train_lstm(self) -> int:
        """Train LSTM Price Predictor"""
        try:
            from services.price_predictor import price_predictor
            
            if price_predictor:
                symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
                samples = 0
                
                for symbol in symbols:
                    for timeframe in ['1h', '4h']:
                        try:
                            await price_predictor._train_model(symbol, timeframe)
                            samples += 1000
                        except:
                            pass
                            
                return samples
                
        except Exception as e:
            logger.error(f"LSTM training error: {e}")
        return 0
        
    async def _get_model_accuracy(self, job_id: str) -> float:
        """Get current model accuracy"""
        try:
            if job_id == 'xgboost_edge':
                data = await self.redis_client.hget('xgboost:metrics', 'accuracy')
                return float(data.decode()) if data else 0.0
            elif job_id == 'learning_engine':
                data = await self.redis_client.get('learning:win_rate')
                return float(data.decode()) if data else 0.5
            elif job_id == 'lstm_predictor':
                data = await self.redis_client.hget('lstm:metrics', 'accuracy')
                return float(data.decode()) if data else 0.0
        except:
            pass
        return 0.0
        
    async def _load_job_states(self):
        """Load job states from Redis"""
        try:
            for job_id in self.jobs:
                data = await self.redis_client.hgetall(f'trainer:job:{job_id}')
                if data:
                    job = self.jobs[job_id]
                    job.last_run = datetime.fromisoformat(data[b'last_run'].decode()) if data.get(b'last_run') else None
                    job.status = data[b'status'].decode() if data.get(b'status') else 'pending'
        except Exception as e:
            logger.debug(f"Load job states error: {e}")
            
    async def _save_job_states(self):
        """Save job states to Redis"""
        try:
            for job_id, job in self.jobs.items():
                await self.redis_client.hset(f'trainer:job:{job_id}', mapping={
                    'last_run': job.last_run.isoformat() if job.last_run else '',
                    'status': job.status
                })
        except Exception as e:
            logger.debug(f"Save job states error: {e}")
            
    async def trigger_training(self, job_id: str) -> bool:
        """Manually trigger a training job"""
        if job_id not in self.jobs:
            return False
            
        job = self.jobs[job_id]
        await self._queue_job(job)
        return True
        
    async def get_status(self) -> Dict:
        """Get trainer status"""
        return {
            'is_running': self.is_running,
            'active_training': self.active_training,
            'jobs': {
                job_id: {
                    'model_name': job.model_name,
                    'interval_hours': job.interval_hours,
                    'last_run': job.last_run.isoformat() if job.last_run else None,
                    'status': job.status,
                    'priority': job.priority,
                    'estimated_duration_minutes': job.estimated_duration_minutes
                }
                for job_id, job in self.jobs.items()
            },
            'recent_history': [
                {
                    'model_name': m.model_name,
                    'completed_at': m.completed_at,
                    'duration_seconds': m.duration_seconds,
                    'improvement': m.improvement,
                    'status': m.status
                }
                for m in self.training_history[-10:]
            ]
        }


# Global instance
model_trainer = ModelTrainer()

