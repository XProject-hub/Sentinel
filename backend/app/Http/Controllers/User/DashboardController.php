<?php

namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\Trade;
use App\Models\Position;
use App\Models\TradingStatistic;
use App\Services\AIStatusService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Carbon\Carbon;

class DashboardController extends Controller
{
    /**
     * Main dashboard data
     */
    public function index(Request $request)
    {
        $user = $request->user();
        
        // Cache dashboard data for 30 seconds
        $cacheKey = "dashboard:{$user->id}";
        
        $data = Cache::remember($cacheKey, 30, function () use ($user) {
            return [
                'balance' => $this->getBalanceData($user),
                'performance' => $this->getPerformanceData($user),
                'positions' => $this->getActivePositions($user),
                'ai_status' => $this->getAIStatus($user),
                'risk_status' => $this->getRiskStatus($user),
                'recent_trades' => $this->getRecentTrades($user),
            ];
        });

        return response()->json([
            'success' => true,
            'data' => $data
        ]);
    }

    /**
     * Dashboard summary (lightweight)
     */
    public function summary(Request $request)
    {
        $user = $request->user();

        $totalBalance = $user->balances()->sum('total_usd_value');
        $todayPnl = $this->getTodayPnl($user);
        $weeklyPnl = $this->getWeeklyPnl($user);
        $activePositions = $user->positions()->count();

        return response()->json([
            'success' => true,
            'data' => [
                'total_balance' => round($totalBalance, 2),
                'today_pnl' => round($todayPnl, 2),
                'today_pnl_percent' => $totalBalance > 0 ? round(($todayPnl / $totalBalance) * 100, 2) : 0,
                'weekly_pnl' => round($weeklyPnl, 2),
                'weekly_pnl_percent' => $totalBalance > 0 ? round(($weeklyPnl / $totalBalance) * 100, 2) : 0,
                'active_positions' => $activePositions,
            ]
        ]);
    }

    /**
     * Performance metrics
     */
    public function performance(Request $request)
    {
        $user = $request->user();
        $period = $request->get('period', '7d');

        $stats = $this->getPerformanceStats($user, $period);

        return response()->json([
            'success' => true,
            'data' => $stats
        ]);
    }

    /**
     * AI system status
     */
    public function aiStatus(Request $request)
    {
        $user = $request->user();
        
        // Get from Redis (real-time AI status)
        $aiStatus = Cache::get("ai_status:{$user->id}", [
            'active' => true,
            'confidence' => 0.75,
            'current_regime' => 'sideways',
            'active_strategy' => 'Grid Master',
            'last_analysis' => now()->subMinutes(2)->toISOString(),
            'insight' => 'Market showing low volatility. Maintaining conservative positions.',
            'next_action' => 'Monitoring for breakout signals.',
        ]);

        return response()->json([
            'success' => true,
            'data' => $aiStatus
        ]);
    }

    /**
     * Risk management status
     */
    public function riskStatus(Request $request)
    {
        $user = $request->user();
        $riskSettings = $user->riskSettings;
        
        $todayLoss = $this->getTodayLoss($user);
        $currentExposure = $this->getCurrentExposure($user);
        
        $status = 'SAFE';
        if ($todayLoss > ($riskSettings->max_loss_per_day ?? 5) * 0.7) {
            $status = 'CAUTION';
        }
        if ($todayLoss >= ($riskSettings->max_loss_per_day ?? 5)) {
            $status = 'STOPPED';
        }

        return response()->json([
            'success' => true,
            'data' => [
                'status' => $status,
                'today_loss_percent' => round($todayLoss, 2),
                'max_loss_allowed' => $riskSettings->max_loss_per_day ?? 5,
                'current_exposure_percent' => round($currentExposure, 2),
                'max_exposure_allowed' => $riskSettings->max_exposure_percent ?? 30,
                'positions_count' => $user->positions()->count(),
                'max_positions' => $riskSettings->max_positions ?? 5,
                'emergency_stop_active' => false,
                'cooldown_active' => false,
            ]
        ]);
    }

    /**
     * Statistics overview
     */
    public function statsOverview(Request $request)
    {
        $user = $request->user();

        $allTimeStats = Trade::where('user_id', $user->id)
            ->where('status', 'closed')
            ->selectRaw('
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            ')
            ->first();

        $winRate = $allTimeStats->total_trades > 0 
            ? ($allTimeStats->winning_trades / $allTimeStats->total_trades) * 100 
            : 0;

        return response()->json([
            'success' => true,
            'data' => [
                'total_trades' => $allTimeStats->total_trades ?? 0,
                'winning_trades' => $allTimeStats->winning_trades ?? 0,
                'losing_trades' => $allTimeStats->losing_trades ?? 0,
                'win_rate' => round($winRate, 2),
                'total_pnl' => round($allTimeStats->total_pnl ?? 0, 2),
                'average_pnl' => round($allTimeStats->avg_pnl ?? 0, 2),
                'best_trade' => round($allTimeStats->best_trade ?? 0, 2),
                'worst_trade' => round($allTimeStats->worst_trade ?? 0, 2),
            ]
        ]);
    }

    // Private helper methods

    private function getBalanceData($user): array
    {
        $totalBalance = $user->balances()->sum('total_usd_value');
        $balanceByAsset = $user->balances()
            ->where('total_usd_value', '>', 0)
            ->orderByDesc('total_usd_value')
            ->limit(5)
            ->get(['asset', 'free_balance', 'total_usd_value']);

        return [
            'total_usd' => round($totalBalance, 2),
            'by_asset' => $balanceByAsset,
        ];
    }

    private function getPerformanceData($user): array
    {
        return [
            'today' => round($this->getTodayPnl($user), 2),
            'week' => round($this->getWeeklyPnl($user), 2),
            'month' => round($this->getMonthlyPnl($user), 2),
            'all_time' => round($this->getAllTimePnl($user), 2),
        ];
    }

    private function getActivePositions($user): array
    {
        return $user->positions()
            ->with('strategy')
            ->get()
            ->map(function ($position) {
                return [
                    'id' => $position->id,
                    'symbol' => $position->symbol,
                    'side' => $position->side,
                    'entry_price' => $position->entry_price,
                    'current_price' => $position->current_price,
                    'quantity' => $position->quantity,
                    'unrealized_pnl' => round($position->unrealized_pnl, 2),
                    'unrealized_pnl_percent' => round($position->unrealized_pnl_percent, 2),
                    'strategy' => $position->strategy?->name,
                ];
            })
            ->toArray();
    }

    private function getAIStatus($user): array
    {
        return Cache::get("ai_status:{$user->id}", [
            'active' => true,
            'confidence' => 0.72,
            'current_regime' => 'sideways',
            'active_strategy' => 'Grid Master',
            'insight' => 'Monitoring market conditions. Low volatility detected.',
        ]);
    }

    private function getRiskStatus($user): array
    {
        $riskSettings = $user->riskSettings;
        $todayLoss = $this->getTodayLoss($user);
        
        return [
            'status' => $todayLoss < ($riskSettings->max_loss_per_day ?? 5) * 0.7 ? 'SAFE' : 'CAUTION',
            'today_loss_percent' => round($todayLoss, 2),
        ];
    }

    private function getRecentTrades($user): array
    {
        return $user->trades()
            ->where('status', 'closed')
            ->orderByDesc('closed_at')
            ->limit(5)
            ->get()
            ->map(function ($trade) {
                return [
                    'id' => $trade->id,
                    'symbol' => $trade->symbol,
                    'side' => $trade->side,
                    'pnl' => round($trade->pnl, 2),
                    'closed_at' => $trade->closed_at->toISOString(),
                ];
            })
            ->toArray();
    }

    private function getTodayPnl($user): float
    {
        return Trade::where('user_id', $user->id)
            ->where('status', 'closed')
            ->whereDate('closed_at', Carbon::today())
            ->sum('pnl') ?? 0;
    }

    private function getWeeklyPnl($user): float
    {
        return Trade::where('user_id', $user->id)
            ->where('status', 'closed')
            ->where('closed_at', '>=', Carbon::now()->subDays(7))
            ->sum('pnl') ?? 0;
    }

    private function getMonthlyPnl($user): float
    {
        return Trade::where('user_id', $user->id)
            ->where('status', 'closed')
            ->where('closed_at', '>=', Carbon::now()->subDays(30))
            ->sum('pnl') ?? 0;
    }

    private function getAllTimePnl($user): float
    {
        return Trade::where('user_id', $user->id)
            ->where('status', 'closed')
            ->sum('pnl') ?? 0;
    }

    private function getTodayLoss($user): float
    {
        $todayPnl = $this->getTodayPnl($user);
        $totalBalance = $user->balances()->sum('total_usd_value');
        
        if ($totalBalance <= 0 || $todayPnl >= 0) return 0;
        
        return abs($todayPnl / $totalBalance) * 100;
    }

    private function getCurrentExposure($user): float
    {
        $totalBalance = $user->balances()->sum('total_usd_value');
        if ($totalBalance <= 0) return 0;
        
        $positionsValue = $user->positions()->sum(\DB::raw('quantity * current_price'));
        return ($positionsValue / $totalBalance) * 100;
    }

    private function getPerformanceStats($user, string $period): array
    {
        $startDate = match($period) {
            '24h' => Carbon::now()->subHours(24),
            '7d' => Carbon::now()->subDays(7),
            '30d' => Carbon::now()->subDays(30),
            '90d' => Carbon::now()->subDays(90),
            default => Carbon::now()->subDays(7),
        };

        $trades = Trade::where('user_id', $user->id)
            ->where('status', 'closed')
            ->where('closed_at', '>=', $startDate)
            ->get();

        $totalTrades = $trades->count();
        $winningTrades = $trades->where('pnl', '>', 0)->count();
        $totalPnl = $trades->sum('pnl');

        return [
            'period' => $period,
            'total_trades' => $totalTrades,
            'winning_trades' => $winningTrades,
            'losing_trades' => $totalTrades - $winningTrades,
            'win_rate' => $totalTrades > 0 ? round(($winningTrades / $totalTrades) * 100, 2) : 0,
            'total_pnl' => round($totalPnl, 2),
            'avg_pnl' => $totalTrades > 0 ? round($totalPnl / $totalTrades, 2) : 0,
        ];
    }
}

