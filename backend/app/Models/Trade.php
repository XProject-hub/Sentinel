<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Concerns\HasUuids;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Trade extends Model
{
    use HasFactory, HasUuids;

    protected $keyType = 'string';
    public $incrementing = false;

    protected $fillable = [
        'user_id',
        'exchange',
        'symbol',
        'side',
        'order_type',
        'quantity',
        'price',
        'filled_quantity',
        'average_price',
        'status',
        'strategy_id',
        'ai_confidence',
        'ai_reasoning',
        'exchange_order_id',
        'fee',
        'fee_asset',
        'pnl',
        'pnl_percent',
        'filled_at',
        'closed_at',
    ];

    protected function casts(): array
    {
        return [
            'quantity' => 'decimal:10',
            'price' => 'decimal:10',
            'filled_quantity' => 'decimal:10',
            'average_price' => 'decimal:10',
            'ai_confidence' => 'decimal:2',
            'fee' => 'decimal:10',
            'pnl' => 'decimal:2',
            'pnl_percent' => 'decimal:4',
            'filled_at' => 'datetime',
            'closed_at' => 'datetime',
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

