<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Concerns\HasUuids;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class RiskSettings extends Model
{
    use HasFactory, HasUuids;

    protected $keyType = 'string';
    public $incrementing = false;

    protected $fillable = [
        'user_id',
        'max_loss_per_trade',
        'max_loss_per_day',
        'max_exposure_percent',
        'max_positions',
        'cooldown_after_loss_minutes',
        'emergency_stop_enabled',
        'emergency_stop_loss_percent',
    ];

    protected function casts(): array
    {
        return [
            'max_loss_per_trade' => 'decimal:2',
            'max_loss_per_day' => 'decimal:2',
            'max_exposure_percent' => 'decimal:2',
            'max_positions' => 'integer',
            'cooldown_after_loss_minutes' => 'integer',
            'emergency_stop_enabled' => 'boolean',
            'emergency_stop_loss_percent' => 'decimal:2',
        ];
    }

    public function user()
    {
        return $this->belongsTo(User::class);
    }
}

