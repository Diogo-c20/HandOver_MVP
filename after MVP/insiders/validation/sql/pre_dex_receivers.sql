-- Template: Pre-DEX Token Receivers
-- Goal: Wallets that received the token from the deployer BEFORE the first DEX liquidity event.
-- Params:
--   {{token_address}} : lowercase ERC-20 contract address
-- Notes:
-- - Replace dataset names if your Dune environment differs.
-- - If `dex.trades` is unavailable, UNION specific DEX trade tables (uniswap_v2_ethereum.trades, uniswap_v3_ethereum.trades, sushiswap_ethereum.trades, etc.).

WITH
-- 1) Resolve deployer address for the token
creator AS (
    -- Option A: contracts.creation_traces
    SELECT LOWER(creator_address) AS deployer
    FROM ethereum.contracts
    WHERE address = LOWER('{{token_address}}')
    LIMIT 1
),
-- 2) First liquidity/trade time on any major DEX for this token
first_liquidity AS (
    SELECT MIN(block_time) AS first_liq_ts
    FROM dex.trades
    WHERE LOWER(contract_address) = LOWER('{{token_address}}')
       OR LOWER(token_a) = LOWER('{{token_address}}')
       OR LOWER(token_b) = LOWER('{{token_address}}')
),
-- 3) Token transfers from deployer before first liquidity
pre_dex_transfers AS (
    SELECT
        LOWER(t."to") AS receiver,
        t.evt_block_time
    FROM erc20_ethereum.evt_Transfer t
    CROSS JOIN creator c
    CROSS JOIN first_liquidity f
    WHERE LOWER(t.contract_address) = LOWER('{{token_address}}')
      AND LOWER(t."from") = c.deployer
      AND t.evt_block_time < f.first_liq_ts
)
SELECT DISTINCT receiver AS wallet
FROM pre_dex_transfers
WHERE receiver IS NOT NULL;
