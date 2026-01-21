<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Concerns\HasUuids;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\Crypt;

class ExchangeConnection extends Model
{
    use HasFactory, HasUuids;

    protected $keyType = 'string';
    public $incrementing = false;

    protected $fillable = [
        'user_id',
        'exchange',
        'name',
        'api_key_encrypted',
        'api_secret_encrypted',
        'is_testnet',
        'is_active',
        'last_sync_at',
        'permissions',
    ];

    protected $hidden = [
        'api_key_encrypted',
        'api_secret_encrypted',
    ];

    protected function casts(): array
    {
        return [
            'is_testnet' => 'boolean',
            'is_active' => 'boolean',
            'last_sync_at' => 'datetime',
            'permissions' => 'array',
        ];
    }

    // Encrypt API key before saving (maps api_key to api_key_encrypted column)
    public function setApiKeyAttribute($value)
    {
        $this->attributes['api_key_encrypted'] = $value ? Crypt::encryptString($value) : null;
    }

    // Decrypt API key when retrieving
    public function getApiKeyAttribute()
    {
        $value = $this->attributes['api_key_encrypted'] ?? null;
        return $value ? Crypt::decryptString($value) : null;
    }

    // Encrypt API secret before saving (maps api_secret to api_secret_encrypted column)
    public function setApiSecretAttribute($value)
    {
        $this->attributes['api_secret_encrypted'] = $value ? Crypt::encryptString($value) : null;
    }

    // Decrypt API secret when retrieving
    public function getApiSecretAttribute()
    {
        $value = $this->attributes['api_secret_encrypted'] ?? null;
        return $value ? Crypt::decryptString($value) : null;
    }

    // Relationships
    public function user()
    {
        return $this->belongsTo(User::class);
    }

    // Get masked API key for display
    public function getMaskedApiKey(): string
    {
        $key = $this->api_key;
        if (!$key) return '';
        return substr($key, 0, 6) . '...' . substr($key, -4);
    }

    // Check if connection is valid
    public function isValid(): bool
    {
        return $this->api_key && $this->api_secret && $this->is_active;
    }
}

