-- Template: Deployer-Funded Early Buyers
-- Goal: Wallets that received native gas token from the deployer shortly BEFORE launch
--       AND were early buyers shortly AFTER launch.
-- Params:
--   {{token_address}}     : lowercase ERC-20 contract address
--   {{hours_pre_liq}}     : hours window before first liquidity (e.g., 24)
--   {{post_liq_minutes}}  : minutes window after first liquidity to call "early" (e.g., 60)
-- Notes:
-- - Adjust DEX fields (trader/buyer/taker) to your schema.

WITH
creator AS (
    SELECT LOWER(creator_address) AS deployer
    FROM ethereum.contracts
    WHERE address = LOWER('{{token_address}}')
    LIMIT 1
),
first_liquidity AS (
    SELECT MIN(block_time) AS first_liq_ts
    FROM dex.trades
    WHERE LOWER(contract_address) = LOWER('{{token_address}}')
       OR LOWER(token_a) = LOWER('{{token_address}}')
       OR LOWER(token_b) = LOWER('{{token_address}}')
),
-- Native coin funding from deployer within pre-liquidity window
native_funding AS (
    SELECT
        LOWER(tx."to")   AS wallet,
        tx.block_time     AS ts,
        tx.value
    FROM ethereum.transactions tx
    CROSS JOIN creator c
    CROSS JOIN first_liquidity f
    WHERE LOWER(tx."from") = c.deployer
      AND tx.value > 0
      AND tx.block_time BETWEEN f.first_liq_ts - INTERVAL '{{hours_pre_liq}} hours' AND f.first_liq_ts
),
-- Early buys in the minutes after first liquidity
early_buys AS (
    SELECT DISTINCT LOWER(trader) AS wallet
    FROM dex.trades d
    CROSS JOIN first_liquidity f
    WHERE (LOWER(d.token_a) = LOWER('{{token_address}}') OR LOWER(d.token_b) = LOWER('{{token_address}}'))
      AND d.block_time BETWEEN f.first_liq_ts AND f.first_liq_ts + INTERVAL '{{post_liq_minutes}} minutes'
      AND d.side = 'buy'  -- Replace with your schema's side indicator
)
SELECT DISTINCT nf.wallet
FROM native_funding nf
JOIN early_buys eb ON eb.wallet = nf.wallet
WHERE nf.wallet IS NOT NULL;
