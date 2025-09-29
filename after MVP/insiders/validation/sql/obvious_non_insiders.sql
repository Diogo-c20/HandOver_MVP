-- Template: Obvious Non-Insiders (Late + CEX-funded)
-- Goal: Wallets whose first acquisition is >= 30 days after launch and are funded primarily by a major CEX.
-- Params:
--   {{token_address}} : lowercase ERC-20 contract
--   {{sample_size}}   : integer limit (e.g., 100)
-- Notes:
-- - Uses labels to identify CEX addresses; adjust label source as needed.

WITH
first_liquidity AS (
    SELECT MIN(block_time) AS first_liq_ts
    FROM dex.trades
    WHERE LOWER(contract_address) = LOWER('{{token_address}}')
       OR LOWER(token_a) = LOWER('{{token_address}}')
       OR LOWER(token_b) = LOWER('{{token_address}}')
),
-- Wallet's first token acquisition time (via transfer in)
first_acquisition AS (
    SELECT LOWER(t."to") AS wallet, MIN(t.evt_block_time) AS first_ts
    FROM erc20_ethereum.evt_Transfer t
    WHERE LOWER(t.contract_address) = LOWER('{{token_address}}')
    GROUP BY 1
),
-- Late adopters: first acquisition occurs >= 30 days after first liquidity
late_adopters AS (
    SELECT fa.wallet, fa.first_ts
    FROM first_acquisition fa
    CROSS JOIN first_liquidity f
    WHERE fa.first_ts >= f.first_liq_ts + INTERVAL '30 days'
),
-- Funding parents before first acquisition
funding_sources AS (
    SELECT DISTINCT LOWER(tx."from") AS parent, LOWER(tx."to") AS wallet
    FROM ethereum.transactions tx
    JOIN late_adopters la ON LOWER(tx."to") = la.wallet AND tx.block_time <= la.first_ts
),
-- Label CEX entities (adjust to your labels dataset)
cex_labels AS (
    SELECT LOWER(address) AS addr
    FROM labels.all
    WHERE label_type ILIKE '%cex%'
)
SELECT DISTINCT fs.wallet
FROM funding_sources fs
JOIN cex_labels cl ON cl.addr = fs.parent
WHERE fs.wallet IS NOT NULL
LIMIT {{sample_size}};
