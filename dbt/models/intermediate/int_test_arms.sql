{#- Deduplicated arms, numbered within each test, with identical arms (natural A/A tests)
    flagged. Two arms are identical when they share a test, the same headline and the same
    eyecatcher_id; both must be non-null, since a missing image id says nothing about whether
    the images matched. Strictly identical arms also share excerpt and lede (null matching
    null), so nothing a visitor could see differs between them. -#}

with packages as (

    select * from {{ ref('stg_packages') }}

),

deduped as (

    -- An exact duplicate repeats every content column; package_id and source_row differ
    -- because they come from the CSV row position.
    select
        *,
        count(*) over (
            partition by clickability_test_id, created_at, updated_at, test_week, headline,
                eyecatcher_id, excerpt, lede, impressions, clicks, significance, first_place,
                winner, source_part
        ) as n_copies
    from packages
    qualify row_number() over (
        partition by clickability_test_id, created_at, updated_at, test_week, headline,
            eyecatcher_id, excerpt, lede, impressions, clicks, significance, first_place,
            winner, source_part
        order by package_id
    ) = 1

),

numbered as (

    select
        *,
        row_number() over (
            partition by clickability_test_id
            order by created_at, source_row, package_id
        ) as arm_number,
        case
            when headline is not null and eyecatcher_id is not null
                then count(*) over (partition by clickability_test_id, headline, eyecatcher_id)
            else 1
        end as n_identical_in_group,
        case
            when headline is not null and eyecatcher_id is not null
                then count(*) over (
                    partition by clickability_test_id, headline, eyecatcher_id, excerpt, lede
                )
            else 1
        end as n_strict_identical_in_group
    from deduped

)

select
    package_id,
    clickability_test_id,
    arm_number,
    created_at,
    test_week,
    headline,
    eyecatcher_id,
    excerpt,
    lede,
    impressions,
    clicks,
    ctr,
    significance,
    first_place,
    winner,
    randomization_risk,
    n_copies - 1                                         as n_duplicates_dropped,
    n_identical_in_group > 1                              as is_identical_arm,
    case when n_identical_in_group > 1 then
        min(arm_number) over (partition by clickability_test_id, headline, eyecatcher_id)
    end                                                  as identical_group_first_arm,
    n_identical_in_group,
    n_strict_identical_in_group > 1                       as is_strict_identical_arm,
    n_strict_identical_in_group,
    source_part
from numbered
