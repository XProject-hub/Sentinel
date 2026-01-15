<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Concerns\HasUuids;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Position extends Model
{
    use HasFactory, HasUuids;

    protected $keyType = 'string';
    public $incrementing = false;

    protected $fillable = [
        'user_id',
        'exchange',
        'symbol',
        'side',
        'entry_price',
        'current_price',
        'quantity',
        'leverage',
        'unrealized_pnl',
        'unrealized_pnl_percent',
        'stop_loss',
        'take_profit',
        'strategy_id',
        'opened_at',
    ];

    protected function casts(): array
    {
        return [
            'entry_price' => 'decimal:10',
            'current_price' => 'decimal:10',
            'quantity' => 'decimal:10',
            'leverage' => 'decimal:2',
            'unrealized_pnl' => 'decimal:2',
            'unrealized_pnl_percent' => 'decimal:4',
            'stop_loss' => 'decimal:10',
            'take_profit' => 'decimal:10',
            'opened_at' => 'datetime',
        ];
    }

    public function user()
    {
        return $this->belongsTo(User::class);
    }

    public function strategy()
    {
        return $this->belongsTo(AIStrategy::class, 'strategy_id');
    }
}

