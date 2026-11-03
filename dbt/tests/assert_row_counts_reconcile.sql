{#- Row counts must reconcile through the layers:
      raw.packages rows = stg_packages rows
      stg_packages rows = int_test_arms rows + duplicates dropped
      distinct tests in raw = fct_tests rows
      sum(fct_tests.n_arms) = int_test_arms rows
    Returns one row per check that fails. -#}

with counts as (
    select
        (select count(*) from {{ source('upworthy', 'packages') }})                         as raw_rows,
        (select count(distinct clickability_test_id) from {{ source('upworthy', 'packages') }}) as raw_tests,
        (select count(*) from {{ ref('stg_packages') }})                                     as stg_rows,
        (select count(*) from {{ ref('int_test_arms') }})                                    as int_rows,
        (select sum(n_duplicates_dropped) from {{ ref('int_test_arms') }})                   as dupes_dropped,
        (select count(*) from {{ ref('fct_tests') }})                                        as fct_tests,
        (select sum(n_arms) from {{ ref('fct_tests') }})                                     as fct_arms
),

checks as (
    select 'raw rows = staging rows' as check_name, raw_rows as expected, stg_rows as actual from counts
    union all
    select 'staging rows = arms + duplicates dropped', stg_rows, int_rows + dupes_dropped from counts
    union all
    select 'raw tests = fct_tests rows', raw_tests, fct_tests from counts
    union all
    select 'sum(n_arms) = arms', int_rows, fct_arms from counts
)

select * from checks where expected is distinct from actual
