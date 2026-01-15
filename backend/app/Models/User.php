<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Concerns\HasUuids;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Foundation\Auth\User as Authenticatable;
use Illuminate\Notifications\Notifiable;
use Tymon\JWTAuth\Contracts\JWTSubject;
use Spatie\Activitylog\Traits\LogsActivity;
use Spatie\Activitylog\LogOptions;

class User extends Authenticatable implements JWTSubject
{
    use HasFactory, Notifiable, HasUuids, LogsActivity;

    protected $keyType = 'string';
    public $incrementing = false;

    protected $fillable = [
        'name',
        'email',
        'password',
        'subscription_tier',
        'subscription_expires_at',
        'is_active',
        'two_factor_enabled',
        'two_factor_secret',
    ];

    protected $hidden = [
        'password',
        'two_factor_secret',
    ];

    protected function casts(): array
    {
        return [
            'email_verified_at' => 'datetime',
            'subscription_expires_at' => 'datetime',
            'password' => 'hashed',
            'is_active' => 'boolean',
            'two_factor_enabled' => 'boolean',
        ];
    }

    public function getActivitylogOptions(): LogOptions
    {
        return LogOptions::defaults()
            ->logOnly(['name', 'email', 'subscription_tier'])
            ->logOnlyDirty();
    }

    // JWT Methods
    public function getJWTIdentifier()
    {
        return $this->getKey();
    }

    public function getJWTCustomClaims()
    {
        return [
            'subscription' => $this->subscription_tier,
        ];
    }

    // Relationships
    public function exchangeConnections()
    {
        return $this->hasMany(ExchangeConnection::class);
    }

    public function balances()
    {
        return $this->hasMany(AccountBalance::class);
    }

    public function trades()
    {
        return $this->hasMany(Trade::class);
    }

    public function positions()
    {
        return $this->hasMany(Position::class);
    }

    public function riskSettings()
    {
        return $this->hasOne(RiskSettings::class);
    }

    public function strategies()
    {
        return $this->hasMany(UserStrategy::class);
    }

    public function notifications()
    {
        return $this->hasMany(Notification::class);
    }

    public function statistics()
    {
        return $this->hasMany(TradingStatistic::class);
    }

    // Helpers
    public function hasActiveSubscription(): bool
    {
        return $this->subscription_tier !== 'free' 
            && $this->subscription_expires_at 
            && $this->subscription_expires_at->isFuture();
    }

    public function getTotalBalance(): float
    {
        return $this->balances()->sum('total_usd_value');
    }

    public function getActivePositionsCount(): int
    {
        return $this->positions()->count();
    }
}

