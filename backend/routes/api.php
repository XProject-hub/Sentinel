<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Auth\AuthController;
use App\Http\Controllers\Auth\TwoFactorController;
use App\Http\Controllers\User\ProfileController;
use App\Http\Controllers\User\DashboardController;
use App\Http\Controllers\Exchange\ExchangeController;
use App\Http\Controllers\Trading\TradeController;
use App\Http\Controllers\Trading\PositionController;
use App\Http\Controllers\Strategy\StrategyController;
use App\Http\Controllers\Risk\RiskController;
use App\Http\Controllers\Subscription\SubscriptionController;
use App\Http\Controllers\Notification\NotificationController;

/*
|--------------------------------------------------------------------------
| SENTINEL AI - API Routes
|--------------------------------------------------------------------------
*/

// ============================================
// PUBLIC ROUTES
// ============================================

Route::prefix('auth')->group(function () {
    Route::post('/register', [AuthController::class, 'register']);
    Route::post('/login', [AuthController::class, 'login']);
    Route::post('/forgot-password', [AuthController::class, 'forgotPassword']);
    Route::post('/reset-password', [AuthController::class, 'resetPassword']);
    Route::post('/verify-email/{token}', [AuthController::class, 'verifyEmail']);
});

// Subscription plans (public)
Route::get('/plans', [SubscriptionController::class, 'plans']);

// ============================================
// AUTHENTICATED ROUTES
// ============================================

Route::middleware(['auth:api'])->group(function () {

    // --------------------------------------------
    // AUTH & 2FA
    // --------------------------------------------
    Route::prefix('auth')->group(function () {
        Route::post('/logout', [AuthController::class, 'logout']);
        Route::post('/refresh', [AuthController::class, 'refresh']);
        Route::get('/me', [AuthController::class, 'me']);
        
        // Two-Factor Authentication
        Route::post('/2fa/enable', [TwoFactorController::class, 'enable']);
        Route::post('/2fa/verify', [TwoFactorController::class, 'verify']);
        Route::post('/2fa/disable', [TwoFactorController::class, 'disable']);
    });

    // --------------------------------------------
    // USER PROFILE
    // --------------------------------------------
    Route::prefix('profile')->group(function () {
        Route::get('/', [ProfileController::class, 'show']);
        Route::put('/', [ProfileController::class, 'update']);
        Route::put('/password', [ProfileController::class, 'updatePassword']);
        Route::delete('/', [ProfileController::class, 'delete']);
    });

    // --------------------------------------------
    // DASHBOARD (Main data endpoint)
    // --------------------------------------------
    Route::prefix('dashboard')->group(function () {
        Route::get('/', [DashboardController::class, 'index']);
        Route::get('/summary', [DashboardController::class, 'summary']);
        Route::get('/performance', [DashboardController::class, 'performance']);
        Route::get('/ai-status', [DashboardController::class, 'aiStatus']);
        Route::get('/risk-status', [DashboardController::class, 'riskStatus']);
    });

    // --------------------------------------------
    // EXCHANGE CONNECTIONS
    // --------------------------------------------
    Route::prefix('exchanges')->group(function () {
        Route::get('/', [ExchangeController::class, 'index']);
        Route::post('/', [ExchangeController::class, 'store']);
        Route::get('/{id}', [ExchangeController::class, 'show']);
        Route::put('/{id}', [ExchangeController::class, 'update']);
        Route::delete('/{id}', [ExchangeController::class, 'destroy']);
        Route::post('/{id}/sync', [ExchangeController::class, 'sync']);
        Route::get('/{id}/balances', [ExchangeController::class, 'balances']);
    });

    // --------------------------------------------
    // TRADING
    // --------------------------------------------
    Route::prefix('trades')->group(function () {
        Route::get('/', [TradeController::class, 'index']);
        Route::get('/history', [TradeController::class, 'history']);
        Route::get('/active', [TradeController::class, 'active']);
        Route::get('/{id}', [TradeController::class, 'show']);
        Route::post('/{id}/close', [TradeController::class, 'close']);
    });

    Route::prefix('positions')->group(function () {
        Route::get('/', [PositionController::class, 'index']);
        Route::get('/{id}', [PositionController::class, 'show']);
        Route::put('/{id}/stop-loss', [PositionController::class, 'updateStopLoss']);
        Route::put('/{id}/take-profit', [PositionController::class, 'updateTakeProfit']);
        Route::post('/{id}/close', [PositionController::class, 'close']);
        Route::post('/close-all', [PositionController::class, 'closeAll']);
    });

    // --------------------------------------------
    // AI STRATEGIES
    // --------------------------------------------
    Route::prefix('strategies')->group(function () {
        Route::get('/', [StrategyController::class, 'index']);
        Route::get('/available', [StrategyController::class, 'available']);
        Route::get('/active', [StrategyController::class, 'active']);
        Route::post('/assign', [StrategyController::class, 'assign']);
        Route::put('/{id}', [StrategyController::class, 'update']);
        Route::delete('/{id}', [StrategyController::class, 'remove']);
        Route::get('/{id}/performance', [StrategyController::class, 'performance']);
    });

    // --------------------------------------------
    // RISK MANAGEMENT
    // --------------------------------------------
    Route::prefix('risk')->group(function () {
        Route::get('/', [RiskController::class, 'index']);
        Route::put('/', [RiskController::class, 'update']);
        Route::get('/events', [RiskController::class, 'events']);
        Route::post('/emergency-stop', [RiskController::class, 'emergencyStop']);
        Route::post('/resume', [RiskController::class, 'resume']);
    });

    // --------------------------------------------
    // SUBSCRIPTIONS & BILLING
    // --------------------------------------------
    Route::prefix('subscription')->group(function () {
        Route::get('/', [SubscriptionController::class, 'current']);
        Route::post('/subscribe', [SubscriptionController::class, 'subscribe']);
        Route::post('/cancel', [SubscriptionController::class, 'cancel']);
        Route::get('/invoices', [SubscriptionController::class, 'invoices']);
        Route::post('/webhook', [SubscriptionController::class, 'webhook'])->withoutMiddleware(['auth:api']);
    });

    // --------------------------------------------
    // NOTIFICATIONS
    // --------------------------------------------
    Route::prefix('notifications')->group(function () {
        Route::get('/', [NotificationController::class, 'index']);
        Route::get('/unread', [NotificationController::class, 'unread']);
        Route::put('/{id}/read', [NotificationController::class, 'markAsRead']);
        Route::put('/read-all', [NotificationController::class, 'markAllAsRead']);
        Route::delete('/{id}', [NotificationController::class, 'destroy']);
    });

    // --------------------------------------------
    // STATISTICS
    // --------------------------------------------
    Route::prefix('stats')->group(function () {
        Route::get('/overview', [DashboardController::class, 'statsOverview']);
        Route::get('/daily', [DashboardController::class, 'dailyStats']);
        Route::get('/weekly', [DashboardController::class, 'weeklyStats']);
        Route::get('/monthly', [DashboardController::class, 'monthlyStats']);
    });
});

