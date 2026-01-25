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

    // Use virtual attributes for api_key/api_secret that map to encrypted columns
    protected $fillable = [
        'user_id',
        'exchange',
        'name',
        'api_key',           // Virtual - triggers mutator
        'api_secret',        // Virtual - triggers mutator
        'is_testnet',
        'is_active',
        'last_sync_at',
        'permissions',
        'region',            // For Bybit regional endpoints: EU, NL, TR, etc.
    ];

    protected $hidden = [
        'api_key_encrypted',
        'api_secret_encrypted',
    ];
    
    // Tell Laravel these are the actual DB columns
    protected $appends = [];

    protected function casts(): array
    {
        return [
            'is_testnet' => 'boolean',
            'is_active' => 'boolean',
            'last_sync_at' => 'datetime',
            'permissions' => 'array',
        ];
    }

    // Encrypt API key before saving - saves to api_key_encrypted column
    public function setApiKeyAttribute($value)
    {
        $this->attributes['api_key_encrypted'] = $value ? Crypt::encryptString($value) : null;
    }

    // Decrypt API key when retrieving from api_key_encrypted column
    public function getApiKeyAttribute()
    {
        $value = $this->attributes['api_key_encrypted'] ?? null;
        if (!$value) return null;
        try {
            return Crypt::decryptString($value);
        } catch (\Exception $e) {
            return null;
        }
    }

    // Encrypt API secret before saving - saves to api_secret_encrypted column
    public function setApiSecretAttribute($value)
    {
        $this->attributes['api_secret_encrypted'] = $value ? Crypt::encryptString($value) : null;
    }

    // Decrypt API secret when retrieving from api_secret_encrypted column
    public function getApiSecretAttribute()
    {
        $value = $this->attributes['api_secret_encrypted'] ?? null;
        if (!$value) return null;
        try {
            return Crypt::decryptString($value);
        } catch (\Exception $e) {
            return null;
        }
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

