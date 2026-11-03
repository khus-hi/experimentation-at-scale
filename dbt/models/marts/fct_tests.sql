{#- One row per A/B test. -#}

with arms as (

    select * from {{ ref('int_test_arms') }}

),

winners as (

    select
        clickability_test_id,
        count(*)                                                as n_declared_winners,
        min(arm_number)                                         as winner_arm_number,
        arg_min(package_id, arm_number)                         as winner_package_id,
        arg_min(headline, arm_number)                           as winner_headline,
        arg_min(ctr, arm_number)                                as winner_ctr
    from arms
    where winner
    group by 1

),

tests as (

    select
        clickability_test_id,
        min(source_part)                                        as source_part,
        min(created_at)::date                                   as start_date,
        min(created_at)                                         as first_created_at,
        min(test_week)                                          as test_week,
        count(*)                                                as n_arms,
        sum(impressions)                                        as total_impressions,
        sum(clicks)                                             as total_clicks,
        sum(impressions)::double / count(*)                     as impressions_per_arm,
        min(impressions)                                        as min_arm_impressions,
        max(impressions)                                        as max_arm_impressions,
        max(ctr)                                                as best_ctr,
        min(ctr)                                                as worst_ctr,
        arg_max(arm_number, ctr)                                as best_ctr_arm_number,
        bool_or(randomization_risk)                             as randomization_risk,
        count(*) filter (where is_identical_arm)                as n_identical_arms,
        sum(n_duplicates_dropped)                               as n_duplicates_dropped
    from arms
    group by 1

),

pairs as (

    select clickability_test_id, count(*) as n_aa_pairs
    from {{ ref('fct_aa_pairs') }}
    group by 1

)

select
    t.clickability_test_id,
    t.source_part,
    t.start_date,
    t.first_created_at,
    t.test_week,
    t.n_arms,
    t.total_impressions,
    t.total_clicks,
    t.impressions_per_arm,
    t.min_arm_impressions,
    t.max_arm_impressions,
    t.best_ctr,
    t.worst_ctr,
    t.best_ctr_arm_number,
    coalesce(w.n_declared_winners, 0)                           as n_declared_winners,
    w.winner_arm_number,
    w.winner_package_id,
    w.winner_headline,
    w.winner_ctr,
    t.randomization_risk,
    t.n_identical_arms > 0                                      as has_aa_pair,
    t.n_identical_arms,
    coalesce(p.n_aa_pairs, 0)                                   as n_aa_pairs,
    t.n_duplicates_dropped
from tests as t
left join winners as w using (clickability_test_id)
left join pairs as p using (clickability_test_id)
