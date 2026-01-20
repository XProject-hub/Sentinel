<?php

namespace App\Http\Controllers\Exchange;

use App\Http\Controllers\Controller;
use App\Models\ExchangeConnection;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class ExchangeController extends Controller
{
    /**
     * List all exchange connections for the authenticated user
     */
    public function index(Request $request)
    {
        $connections = $request->user()->exchangeConnections()
            ->select('id', 'exchange', 'name', 'is_testnet', 'is_active', 'last_sync_at', 'created_at')
            ->get()
            ->map(function ($conn) {
                return [
                    'id' => $conn->id,
                    'exchange' => $conn->exchange,
                    'name' => $conn->name,
                    'is_testnet' => $conn->is_testnet,
                    'is_active' => $conn->is_active,
                    'last_sync_at' => $conn->last_sync_at,
                    'created_at' => $conn->created_at,
                    'api_key_masked' => $conn->getMaskedApiKey(),
                ];
            });

        return response()->json([
            'success' => true,
            'data' => $connections,
        ]);
    }

    /**
     * Create a new exchange connection
     */
    public function store(Request $request)
    {
        $validated = $request->validate([
            'exchange' => 'required|string|in:bybit,binance,okx',
            'name' => 'required|string|max:50',
            'api_key' => 'required|string|min:10',
            'api_secret' => 'required|string|min:10',
            'is_testnet' => 'boolean',
        ]);

        // Check if user already has a connection for this exchange
        $existing = $request->user()->exchangeConnections()
            ->where('exchange', $validated['exchange'])
            ->first();

        if ($existing) {
            return response()->json([
                'success' => false,
                'message' => 'You already have a connection for this exchange. Please update or delete the existing one.',
            ], 422);
        }

        // Verify API credentials before saving
        $verification = $this->verifyCredentials(
            $validated['exchange'],
            $validated['api_key'],
            $validated['api_secret'],
            $validated['is_testnet'] ?? false
        );

        if (!$verification['valid']) {
            return response()->json([
                'success' => false,
                'message' => 'Invalid API credentials: ' . $verification['error'],
            ], 422);
        }

        // Create the connection
        $connection = $request->user()->exchangeConnections()->create([
            'exchange' => $validated['exchange'],
            'name' => $validated['name'],
            'api_key' => $validated['api_key'],
            'api_secret' => $validated['api_secret'],
            'is_testnet' => $validated['is_testnet'] ?? false,
            'is_active' => true,
            'permissions' => $verification['permissions'] ?? [],
        ]);

        // Sync to AI services
        $this->syncToAiServices($request->user()->id, $connection);

        Log::info('Exchange connection created', [
            'user_id' => $request->user()->id,
            'exchange' => $validated['exchange'],
        ]);

        return response()->json([
            'success' => true,
            'message' => 'Exchange connected successfully',
            'data' => [
                'id' => $connection->id,
                'exchange' => $connection->exchange,
                'name' => $connection->name,
                'is_testnet' => $connection->is_testnet,
                'is_active' => $connection->is_active,
                'api_key_masked' => $connection->getMaskedApiKey(),
                'balance' => $verification['balance'] ?? null,
            ],
        ], 201);
    }

    /**
     * Show a specific exchange connection
     */
    public function show(Request $request, string $id)
    {
        $connection = $request->user()->exchangeConnections()->findOrFail($id);

        return response()->json([
            'success' => true,
            'data' => [
                'id' => $connection->id,
                'exchange' => $connection->exchange,
                'name' => $connection->name,
                'is_testnet' => $connection->is_testnet,
                'is_active' => $connection->is_active,
                'last_sync_at' => $connection->last_sync_at,
                'api_key_masked' => $connection->getMaskedApiKey(),
                'permissions' => $connection->permissions,
            ],
        ]);
    }

    /**
     * Update an exchange connection
     */
    public function update(Request $request, string $id)
    {
        $connection = $request->user()->exchangeConnections()->findOrFail($id);

        $validated = $request->validate([
            'name' => 'sometimes|string|max:50',
            'api_key' => 'sometimes|string|min:10',
            'api_secret' => 'sometimes|string|min:10',
            'is_active' => 'sometimes|boolean',
        ]);

        // If updating credentials, verify them first
        if (isset($validated['api_key']) || isset($validated['api_secret'])) {
            $apiKey = $validated['api_key'] ?? $connection->api_key;
            $apiSecret = $validated['api_secret'] ?? $connection->api_secret;

            $verification = $this->verifyCredentials(
                $connection->exchange,
                $apiKey,
                $apiSecret,
                $connection->is_testnet
            );

            if (!$verification['valid']) {
                return response()->json([
                    'success' => false,
                    'message' => 'Invalid API credentials: ' . $verification['error'],
                ], 422);
            }

            $validated['permissions'] = $verification['permissions'] ?? [];
        }

        $connection->update($validated);

        // Sync to AI services
        $this->syncToAiServices($request->user()->id, $connection);

        return response()->json([
            'success' => true,
            'message' => 'Exchange connection updated',
            'data' => [
                'id' => $connection->id,
                'exchange' => $connection->exchange,
                'name' => $connection->name,
                'is_active' => $connection->is_active,
                'api_key_masked' => $connection->getMaskedApiKey(),
            ],
        ]);
    }

    /**
     * Delete an exchange connection
     */
    public function destroy(Request $request, string $id)
    {
        $connection = $request->user()->exchangeConnections()->findOrFail($id);
        
        // Remove from AI services
        $this->removeFromAiServices($request->user()->id, $connection->exchange);
        
        $connection->delete();

        return response()->json([
            'success' => true,
            'message' => 'Exchange connection deleted',
        ]);
    }

    /**
     * Sync exchange data (balances, positions)
     */
    public function sync(Request $request, string $id)
    {
        $connection = $request->user()->exchangeConnections()->findOrFail($id);

        if (!$connection->is_active) {
            return response()->json([
                'success' => false,
                'message' => 'Connection is not active',
            ], 422);
        }

        // Trigger sync in AI services
        try {
            $response = Http::timeout(30)->post(
                'http://ai-services:8000/exchange/sync',
                [
                    'user_id' => $request->user()->id,
                    'exchange' => $connection->exchange,
                ]
            );

            if ($response->successful()) {
                $connection->update(['last_sync_at' => now()]);

                return response()->json([
                    'success' => true,
                    'message' => 'Sync completed',
                    'data' => $response->json(),
                ]);
            }

            return response()->json([
                'success' => false,
                'message' => 'Sync failed: ' . $response->body(),
            ], 500);
        } catch (\Exception $e) {
            Log::error('Exchange sync failed', [
                'user_id' => $request->user()->id,
                'exchange' => $connection->exchange,
                'error' => $e->getMessage(),
            ]);

            return response()->json([
                'success' => false,
                'message' => 'Sync failed: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Get balances for an exchange connection
     */
    public function balances(Request $request, string $id)
    {
        $connection = $request->user()->exchangeConnections()->findOrFail($id);

        if (!$connection->is_active) {
            return response()->json([
                'success' => false,
                'message' => 'Connection is not active',
            ], 422);
        }

        try {
            $response = Http::timeout(15)->get(
                'http://ai-services:8000/exchange/balance',
                [
                    'user_id' => $request->user()->id,
                    'exchange' => $connection->exchange,
                ]
            );

            if ($response->successful()) {
                return response()->json([
                    'success' => true,
                    'data' => $response->json(),
                ]);
            }

            return response()->json([
                'success' => false,
                'message' => 'Failed to get balances',
            ], 500);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Failed to get balances: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Verify API credentials with the exchange
     */
    private function verifyCredentials(string $exchange, string $apiKey, string $apiSecret, bool $isTestnet): array
    {
        try {
            $response = Http::timeout(15)->post(
                'http://ai-services:8000/exchange/verify-credentials',
                [
                    'exchange' => $exchange,
                    'api_key' => $apiKey,
                    'api_secret' => $apiSecret,
                    'is_testnet' => $isTestnet,
                ]
            );

            if ($response->successful()) {
                $data = $response->json();
                return [
                    'valid' => $data['valid'] ?? false,
                    'error' => $data['error'] ?? null,
                    'permissions' => $data['permissions'] ?? [],
                    'balance' => $data['balance'] ?? null,
                ];
            }

            return [
                'valid' => false,
                'error' => 'Failed to verify credentials with exchange',
            ];
        } catch (\Exception $e) {
            Log::error('Credential verification failed', [
                'exchange' => $exchange,
                'error' => $e->getMessage(),
            ]);

            return [
                'valid' => false,
                'error' => 'Connection error: ' . $e->getMessage(),
            ];
        }
    }

    /**
     * Sync credentials to AI services
     */
    private function syncToAiServices(string $userId, ExchangeConnection $connection): void
    {
        try {
            Http::timeout(10)->post(
                'http://ai-services:8000/exchange/set-credentials',
                [
                    'user_id' => $userId,
                    'exchange' => $connection->exchange,
                    'api_key' => $connection->api_key,
                    'api_secret' => $connection->api_secret,
                    'is_testnet' => $connection->is_testnet,
                    'is_active' => $connection->is_active,
                ]
            );
        } catch (\Exception $e) {
            Log::error('Failed to sync credentials to AI services', [
                'user_id' => $userId,
                'error' => $e->getMessage(),
            ]);
        }
    }

    /**
     * Remove credentials from AI services
     */
    private function removeFromAiServices(string $userId, string $exchange): void
    {
        try {
            Http::timeout(10)->delete(
                'http://ai-services:8000/exchange/remove-credentials',
                [
                    'user_id' => $userId,
                    'exchange' => $exchange,
                ]
            );
        } catch (\Exception $e) {
            Log::error('Failed to remove credentials from AI services', [
                'user_id' => $userId,
                'error' => $e->getMessage(),
            ]);
        }
    }
}

