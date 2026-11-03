{#- Every pair of identical arms (same test, headline and eyecatcher_id): a natural A/A test.
    A group of k identical arms gives k*(k-1)/2 pairs. -#}

with identical as (

    select * from {{ ref('int_test_arms') }}
    where is_identical_arm

)

select
    a.clickability_test_id || ':' || a.arm_number || '-' || b.arm_number    as aa_pair_id,
    a.clickability_test_id,
    a.headline,
    a.eyecatcher_id,
    a.arm_number                                                           as arm_a_number,
    b.arm_number                                                           as arm_b_number,
    a.package_id                                                           as package_a_id,
    b.package_id                                                           as package_b_id,
    a.impressions                                                          as impressions_a,
    b.impressions                                                          as impressions_b,
    a.clicks                                                               as clicks_a,
    b.clicks                                                               as clicks_b,
    a.ctr                                                                  as ctr_a,
    b.ctr                                                                  as ctr_b,
    a.n_identical_in_group                                                 as group_size,
    a.randomization_risk or b.randomization_risk                          as randomization_risk,
    a.source_part
from identical as a
join identical as b
    on  a.clickability_test_id = b.clickability_test_id
    and a.headline = b.headline
    and a.eyecatcher_id = b.eyecatcher_id
    and a.arm_number < b.arm_number
