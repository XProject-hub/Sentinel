"""
SENTINEL AI - XGBoost Edge Classifier
Fast ML model for trade signal classification

This replaces rule-based edge estimation with LEARNED classification:
- Trained on historical trade outcomes
- Classifies: BUY, SELL, HOLD
- Returns probability (confidence)
- Retrained periodically (every 6-12 hours)

XGBoost is chosen because:
- Fast inference (milliseconds)
- Handles missing data
- Interpretable (feature importance)
- Works great on tabular data
- No GPU required
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from loguru import logger
import redis.asyncio as redis

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed - classifier will use fallback")

from config import settings

# Import training data manager for quality-filtered data
try:
    from services.training_data_manager import training_data_manager
    TRAINING_MANAGER_AVAILABLE = True
except ImportError:
    TRAINING_MANAGER_AVAILABLE = False


@dataclass
class ClassificationResult:
    """Result of trade signal classification"""
    symbol: str
    timestamp: str
    
    # Classification
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-100
    
    # Probabilities
    buy_prob: float
    sell_prob: float
    hold_prob: float
    
    # Feature importance (top 5)
    top_features: List[Tuple[str, float]]
    
    # Model info
    model_version: str
    training_accuracy: float


class XGBoostClassifier:
    """
    XGBoost-based Trade Signal Classifier
    
    Features used:
    - Price momentum (1h, 4h, 24h)
    - Technical indicators (RSI, MACD, BB)
    - Volume profile
    - Regime information
    - Sentiment data
    
    Trained on:
    - Historical trade outcomes
    - Labeled by: profitable = BUY, loss = SELL, small = HOLD
    """
    
    # Feature names for interpretability
    FEATURE_NAMES = [
        'price_change_1h', 'price_change_4h', 'price_change_24h',
        'rsi_normalized', 'macd_normalized', 'bb_position', 'atr_normalized',
        'volume_ratio', 'volume_trend',
        'regime_encoded', 'trend_strength', 'volatility_normalized', 'liquidity_normalized',
        'fear_greed_normalized', 'news_sentiment', 'funding_rate_normalized'
    ]
    
    def __init__(self):
        self.redis_client = None
        self.is_running = False
        
        # Model
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Model metadata
        self.model_version = "0.0.0"
        self.training_accuracy = 0.0
        self.last_trained = None
        self.training_samples = 0
        
        # Paths
        self.model_dir = Path("/opt/sentinel/models/xgboost")
        self.model_path = self.model_dir / "edge_classifier.pkl"
        self.scaler_path = self.model_dir / "scaler.pkl"
        
        # Training schedule
        self.training_interval_hours = 12  # Retrain every 12 hours
        self.min_samples_for_training = 100  # Minimum trades needed
        
        # Performance tracking
        self.prediction_history: List[Dict] = []
        self.accuracy_rolling = 0.0
        
    async def initialize(self):
        """Initialize classifier"""
        logger.info("Initializing XGBoost Edge Classifier...")
        
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available - using fallback classifier")
            return
            
        self.redis_client = await redis.from_url(settings.REDIS_URL)
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing model
        await self._load_model()
        
        self.is_running = True
        
        # Start periodic training loop
        asyncio.create_task(self._training_loop())
        
        logger.info(f"XGBoost Classifier initialized - Model version: {self.model_version}")
        
    async def shutdown(self):
        """Cleanup"""
        self.is_running = False
        await self._save_model()
        if self.redis_client:
            await self.redis_client.aclose()
            
    async def classify(self, features: Dict[str, float]) -> ClassificationResult:
        """
        Classify trade signal from features
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            ClassificationResult with signal, confidence, and probabilities
        """
        timestamp = datetime.utcnow().isoformat()
        symbol = features.get('symbol', 'UNKNOWN')
        
        if not XGBOOST_AVAILABLE or self.model is None:
            # Fallback to rule-based classification
            return self._fallback_classify(features, timestamp, symbol)
            
        try:
            # Prepare feature vector
            X = self._prepare_features(features)
            
            if X is None:
                return self._fallback_classify(features, timestamp, symbol)
                
            # Scale features
            if self.scaler:
                X = self.scaler.transform(X.reshape(1, -1))
            else:
                X = X.reshape(1, -1)
                
            # Predict probabilities
            probs = self.model.predict_proba(X)[0]
            
            # Classes: 0=HOLD, 1=BUY, 2=SELL
            hold_prob = probs[0] if len(probs) > 0 else 0.33
            buy_prob = probs[1] if len(probs) > 1 else 0.33
            sell_prob = probs[2] if len(probs) > 2 else 0.33
            
            # Determine signal
            max_prob = max(hold_prob, buy_prob, sell_prob)
            if max_prob == buy_prob:
                signal = 'buy'
            elif max_prob == sell_prob:
                signal = 'sell'
            else:
                signal = 'hold'
                
            confidence = max_prob * 100
            
            # Get feature importance for this prediction
            top_features = self._get_top_features(X[0])
            
            result = ClassificationResult(
                symbol=symbol,
                timestamp=timestamp,
                signal=signal,
                confidence=confidence,
                buy_prob=buy_prob,
                sell_prob=sell_prob,
                hold_prob=hold_prob,
                top_features=top_features,
                model_version=self.model_version,
                training_accuracy=self.training_accuracy
            )
            
            # Store for accuracy tracking
            await self._store_prediction(result, features)
            
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._fallback_classify(features, timestamp, symbol)
            
    def _prepare_features(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Prepare feature vector for model"""
        try:
            X = np.array([
                features.get('price_change_1h', 0),
                features.get('price_change_4h', 0),
                features.get('price_change_24h', 0),
                features.get('rsi_normalized', 0.5),
                features.get('macd_normalized', 0),
                features.get('bb_position', 0.5),
                features.get('atr_normalized', 0),
                features.get('volume_ratio', 1),
                features.get('volume_trend', 0),
                features.get('regime_encoded', 0),
                features.get('trend_strength', 0),
                features.get('volatility_normalized', 0.5),
                features.get('liquidity_normalized', 0.5),
                features.get('fear_greed_normalized', 0.5),
                features.get('news_sentiment', 0),
                features.get('funding_rate_normalized', 0),
            ])
            return X
        except:
            return None
            
    def _fallback_classify(self, features: Dict, timestamp: str, symbol: str) -> ClassificationResult:
        """Rule-based fallback when model not available"""
        # Simple rules based on RSI and momentum
        rsi = features.get('rsi_normalized', 0.5) * 100
        momentum = features.get('price_change_1h', 0)
        trend = features.get('trend_strength', 0)
        
        buy_score = 0
        sell_score = 0
        
        # RSI
        if rsi < 30:
            buy_score += 30
        elif rsi > 70:
            sell_score += 30
            
        # Momentum
        if momentum > 0.5:
            buy_score += 25
        elif momentum < -0.5:
            sell_score += 25
            
        # Trend
        if trend > 0.3:
            buy_score += 20
        elif trend < -0.3:
            sell_score += 20
            
        hold_score = 100 - buy_score - sell_score
        
        # Normalize
        total = buy_score + sell_score + hold_score
        buy_prob = buy_score / total
        sell_prob = sell_score / total
        hold_prob = hold_score / total
        
        if buy_prob > sell_prob and buy_prob > hold_prob:
            signal = 'buy'
            confidence = buy_prob * 100
        elif sell_prob > buy_prob and sell_prob > hold_prob:
            signal = 'sell'
            confidence = sell_prob * 100
        else:
            signal = 'hold'
            confidence = hold_prob * 100
            
        return ClassificationResult(
            symbol=symbol,
            timestamp=timestamp,
            signal=signal,
            confidence=confidence,
            buy_prob=buy_prob,
            sell_prob=sell_prob,
            hold_prob=hold_prob,
            top_features=[('rsi', rsi/100), ('momentum', momentum)],
            model_version='fallback',
            training_accuracy=0.0
        )
        
    def _get_top_features(self, X: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top contributing features for this prediction"""
        if self.model is None:
            return []
            
        try:
            importances = self.model.feature_importances_
            
            # Combine with feature values
            contributions = [(self.FEATURE_NAMES[i], float(importances[i] * abs(X[i]))) 
                           for i in range(min(len(self.FEATURE_NAMES), len(importances)))]
            
            # Sort by contribution
            contributions.sort(key=lambda x: x[1], reverse=True)
            
            return contributions[:top_n]
        except:
            return []
            
    async def _training_loop(self):
        """Periodic training loop"""
        logger.info("Starting XGBoost training loop...")
        
        # Initial delay
        await asyncio.sleep(60)
        
        while self.is_running:
            try:
                # Check if training needed
                should_train = await self._should_train()
                
                if should_train:
                    await self._train_model()
                    
                # Wait for next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(3600)
                
    async def _should_train(self) -> bool:
        """Check if model should be retrained"""
        # No model yet
        if self.model is None:
            return True
            
        # Enough time since last training
        if self.last_trained:
            hours_since = (datetime.utcnow() - self.last_trained).total_seconds() / 3600
            if hours_since < self.training_interval_hours:
                return False
                
        # Check if enough new trades
        try:
            trades_count = await self.redis_client.llen('trades:completed')
            if trades_count < self.min_samples_for_training:
                return False
        except:
            return False
            
        return True
        
    async def _train_model(self):
        """Train or retrain the model with quality-weighted data"""
        logger.info("Starting XGBoost model training (with quality weighting)...")
        
        try:
            # Try to get quality-filtered data first (V3)
            X, y, weights = None, None, None
            
            if TRAINING_MANAGER_AVAILABLE:
                try:
                    X, y, weights = await training_data_manager.get_training_dataset(max_samples=10000)
                    if len(X) > 0:
                        logger.info(f"Using quality-filtered data: {len(X)} samples")
                except Exception as e:
                    logger.warning(f"Quality data not available: {e}")
                    
            # Fallback to raw data if quality data not available
            if X is None or len(X) < self.min_samples_for_training:
                X, y = await self._prepare_training_data()
                weights = None  # No weighting for raw data
                
            if X is None or len(X) < self.min_samples_for_training:
                logger.warning(f"Not enough training data: {len(X) if X is not None else 0}")
                return
            
            # Ensure X and y are numpy arrays with correct types
            X = np.array(X, dtype=np.float32)
            y = np.array(y)
            
            # Convert labels to integers (0, 1, 2) if they're not already
            # Handle various label formats: strings, floats, one-hot, etc.
            if y.ndim > 1:
                # One-hot encoded - convert to class indices
                y = np.argmax(y, axis=1)
            elif y.dtype == object or isinstance(y[0], str):
                # String labels - map to integers
                label_map = {'sell': 0, 'hold': 1, 'buy': 2, '0': 0, '1': 1, '2': 2}
                y = np.array([label_map.get(str(label).lower(), 1) for label in y])
            else:
                # Ensure integer type
                y = y.astype(np.int32)
            
            # Ensure labels are in valid range [0, 2]
            y = np.clip(y, 0, 2)
            
            # Check number of unique classes
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            logger.info(f"Training data: X shape={X.shape}, y unique values={unique_classes}, n_classes={n_classes}")
            
            # Handle binary vs multiclass
            if n_classes < 2:
                logger.warning("Not enough classes for training (need at least 2)")
                return
            elif n_classes == 2:
                # Binary classification - adjust objective
                logger.info("Using binary classification (2 classes)")
                self._use_binary = True
            else:
                # Multiclass (3 classes: sell=0, hold=1, buy=2)
                self._use_binary = False
                
            # Split data (with stratification if possible)
            try:
                if weights is not None:
                    weights = np.array(weights, dtype=np.float32)
                    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                        X, y, weights, test_size=0.2, random_state=42, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    w_train, w_test = None, None
            except ValueError:
                # Stratification failed (not enough samples per class)
                logger.warning("Stratification failed, using random split")
                if weights is not None:
                    weights = np.array(weights, dtype=np.float32)
                    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                        X, y, weights, test_size=0.2, random_state=42
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    w_train, w_test = None, None
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model - handle binary vs multiclass
            # XGBoost 1.6+ moved early_stopping_rounds to constructor
            if getattr(self, '_use_binary', False):
                # Binary classification
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'n_jobs': 4,
                    'random_state': 42
                }
            else:
                # Multiclass classification
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'eval_metric': 'mlogloss',
                    'n_jobs': 4,
                    'random_state': 42
                }
            
            try:
                # Try new XGBoost API (1.6+)
                self.model = xgb.XGBClassifier(**model_params, early_stopping_rounds=10)
                self.model.fit(
                    X_train_scaled, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_test_scaled, y_test)],
                    verbose=False
                )
            except TypeError:
                # Fallback for older XGBoost versions
                logger.info("Using legacy XGBoost API")
                self.model = xgb.XGBClassifier(**model_params)
                self.model.fit(
                    X_train_scaled, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_test_scaled, y_test)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            self.training_accuracy = accuracy_score(y_test, y_pred)
            
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Update metadata
            self.training_samples = len(X)
            self.last_trained = datetime.utcnow()
            
            # Increment version
            version_parts = self.model_version.split('.')
            new_minor = int(version_parts[2]) + 1 if len(version_parts) > 2 else 1
            self.model_version = f"1.0.{new_minor}"
            
            # Save model
            await self._save_model()
            
            logger.info(f"XGBoost training complete - v{self.model_version}")
            logger.info(f"Accuracy: {self.training_accuracy:.2%}, Precision: {precision:.2%}, "
                       f"Recall: {recall:.2%}, F1: {f1:.2%}")
            logger.info(f"Trained on {self.training_samples} samples")
            
            # Store training metrics
            await self.redis_client.hset('xgboost:metrics', mapping={
                'version': self.model_version,
                'accuracy': str(self.training_accuracy),
                'precision': str(precision),
                'recall': str(recall),
                'f1': str(f1),
                'samples': str(self.training_samples),
                'trained_at': self.last_trained.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            
    async def _prepare_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from trade history"""
        try:
            # Get completed trades
            trades_data = await self.redis_client.lrange('trades:completed', 0, 9999)
            
            if not trades_data:
                return None, None
                
            X_list = []
            y_list = []
            
            for trade_json in trades_data:
                trade = json.loads(trade_json)
                
                # Skip if missing data
                if 'pnl_percent' not in trade:
                    continue
                    
                # Create feature vector from trade entry data
                X = [
                    trade.get('price_change_1h', 0),
                    trade.get('price_change_4h', 0),
                    trade.get('price_change_24h', 0),
                    trade.get('rsi_normalized', 0.5),
                    trade.get('macd_normalized', 0),
                    trade.get('bb_position', 0.5),
                    trade.get('atr_normalized', 0),
                    trade.get('volume_ratio', 1),
                    trade.get('volume_trend', 0),
                    trade.get('regime_encoded', 0),
                    trade.get('trend_strength', 0),
                    trade.get('volatility_normalized', 0.5),
                    trade.get('liquidity_normalized', 0.5),
                    trade.get('fear_greed_normalized', 0.5),
                    trade.get('news_sentiment', 0),
                    trade.get('funding_rate_normalized', 0),
                ]
                
                # Label based on outcome
                pnl = trade.get('pnl_percent', 0)
                if pnl > 0.5:  # Profitable
                    y = 1  # BUY was correct
                elif pnl < -0.5:  # Loss
                    y = 2  # Should have SOLD
                else:
                    y = 0  # HOLD was appropriate
                    
                X_list.append(X)
                y_list.append(y)
                
            if not X_list:
                return None, None
                
            return np.array(X_list), np.array(y_list)
            
        except Exception as e:
            logger.error(f"Training data preparation error: {e}")
            return None, None
            
    async def _store_prediction(self, result: ClassificationResult, features: Dict):
        """Store prediction for later accuracy tracking"""
        try:
            prediction = {
                'timestamp': result.timestamp,
                'symbol': result.symbol,
                'signal': result.signal,
                'confidence': result.confidence,
                'features': features,
                'verified': False,
                'actual_outcome': None
            }
            
            await self.redis_client.lpush('xgboost:predictions', json.dumps(prediction))
            await self.redis_client.ltrim('xgboost:predictions', 0, 9999)
            
        except:
            pass
            
    async def _load_model(self):
        """Load model from disk"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.model_version = data['version']
                    self.training_accuracy = data['accuracy']
                    self.training_samples = data['samples']
                    self.last_trained = data.get('last_trained')
                    
            if self.scaler_path.exists():
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    
            logger.info(f"Loaded XGBoost model v{self.model_version}")
            
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            
    async def _save_model(self):
        """Save model to disk"""
        if self.model is None:
            return
            
        try:
            data = {
                'model': self.model,
                'version': self.model_version,
                'accuracy': self.training_accuracy,
                'samples': self.training_samples,
                'last_trained': self.last_trained
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
                
            if self.scaler:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                    
            logger.info(f"Saved XGBoost model v{self.model_version}")
            
        except Exception as e:
            logger.error(f"Model save error: {e}")
            
    async def get_stats(self) -> Dict:
        """Get classifier statistics"""
        return {
            'model_version': self.model_version,
            'training_accuracy': self.training_accuracy,
            'training_samples': self.training_samples,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'is_available': XGBOOST_AVAILABLE and self.model is not None,
            'feature_count': len(self.FEATURE_NAMES)
        }


# Global instance
xgboost_classifier = XGBoostClassifier()

