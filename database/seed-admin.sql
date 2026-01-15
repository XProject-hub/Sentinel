-- ============================================
-- SENTINEL AI - Create Admin Account
-- Run this after first deployment
-- ============================================

-- Create admin user
-- Password: SentinelAdmin2026! (hashed with bcrypt)
INSERT INTO users (
    id,
    email,
    password,
    name,
    subscription_tier,
    subscription_expires_at,
    is_active,
    email_verified_at,
    created_at,
    updated_at
) VALUES (
    uuid_generate_v4(),
    'admin@sentinel.ai',
    '$2y$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4d.Q1z5u3xdWYKXe',
    'Admin',
    'enterprise',
    '2030-12-31 23:59:59',
    true,
    NOW(),
    NOW(),
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Create default risk settings for admin
INSERT INTO risk_settings (
    id,
    user_id,
    max_loss_per_trade,
    max_loss_per_day,
    max_exposure_percent,
    max_positions,
    cooldown_after_loss_minutes,
    emergency_stop_enabled,
    emergency_stop_loss_percent,
    created_at,
    updated_at
) 
SELECT 
    uuid_generate_v4(),
    id,
    5.00,
    10.00,
    50.00,
    20,
    15,
    true,
    15.00,
    NOW(),
    NOW()
FROM users 
WHERE email = 'admin@sentinel.ai'
AND NOT EXISTS (
    SELECT 1 FROM risk_settings WHERE user_id = users.id
);

-- Grant admin role (if using role system)
-- INSERT INTO user_roles ...

SELECT 'Admin account created!' as status;
SELECT email, name, subscription_tier FROM users WHERE email = 'admin@sentinel.ai';

