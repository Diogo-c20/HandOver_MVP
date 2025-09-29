-- Dune 프리필터 SQL 템플릿
-- 주의: 기존 search_result_cte는 그대로 사용하십시오.

WITH tokens AS (
  SELECT chain, token_address
  FROM {{ search_result_cte }}
),
dex_stats AS (
  SELECT t.chain, t.token_address,
         SUM(v.volume_usd_24h) AS vol24h_usd,
         SUM(v.trade_count_24h) AS trades24h,
         COUNT(DISTINCT v.trader_24h) AS unique_traders24h
  FROM {{ dex_trades_24h_view }} v
  JOIN tokens t ON v.chain = t.chain AND v.token = t.token_address
  WHERE (
    v.base_symbol IN {{ base_symbols }}
    OR v.base_token_address IN {{ base_addresses_by_chain }}
  )
  GROUP BY 1,2
),
liq AS (
  SELECT t.chain, t.token_address,
         SUM(p.pool_tvl_usd) AS liq_usd
  FROM {{ dex_pools_view }} p
  JOIN tokens t ON p.chain = t.chain AND p.token = t.token_address
  WHERE (
    p.base_symbol IN {{ base_symbols }}
    OR p.base_token_address IN {{ base_addresses_by_chain }}
  )
  GROUP BY 1,2
),
meta AS (
  SELECT t.chain, t.token_address,
         m.mcap_usd,
         DATE_DIFF('day', m.first_swap_ts, NOW()) AS age_d
  FROM {{ token_meta_view }} m
  JOIN tokens t ON m.chain = t.chain AND m.token = t.token_address
)
SELECT
  liq.chain,
  liq.token_address,
  liq.liq_usd,
  ds.vol24h_usd,
  ds.trades24h,
  ds.unique_traders24h,
  mt.mcap_usd,
  mt.age_d
FROM liq
JOIN dex_stats ds USING (chain, token_address)
JOIN meta mt USING (chain, token_address)
WHERE liq_usd              >= {{ min_liq_usd }}
  AND ds.vol24h_usd        >= {{ min_vol24h_usd }}
  AND ds.trades24h         >= {{ min_trades24h }}
  AND ds.unique_traders24h >= {{ min_unique_traders }}
  AND mt.mcap_usd BETWEEN {{ min_mcap }} AND {{ max_mcap }}
  AND mt.age_d             >= {{ min_age_d }}
  AND {{ exclude_base_assets_condition }}
ORDER BY ds.vol24h_usd DESC
LIMIT {{ max_candidates }};

