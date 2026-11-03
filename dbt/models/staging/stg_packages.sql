{#- One row per package (arm). Types cleaned, text trimmed, nothing filtered. -#}
{%- set cols = raw_column_names() -%}
{%- set risk_col = archive_risk_column() -%}

with source as (

    select * from {{ source('upworthy', 'packages') }}

),

typed as (

    select
        {% if 'column00' in cols -%}
        source_part || ':' || cast(column00 as varchar)                as package_id,
        try_cast(column00 as bigint)                                   as source_row,
        {%- else -%}
        source_part || ':' || cast(row_number() over () - 1 as varchar) as package_id,
        cast(null as bigint)                                           as source_row,
        {%- endif %}
        {{ clean_text('clickability_test_id') }}                       as clickability_test_id,
        try_cast(created_at as timestamp)                              as created_at,
        try_cast(updated_at as timestamp)                              as updated_at,
        try_cast(test_week as integer)                                 as test_week,
        {{ clean_text('headline') }}                                   as headline,
        {{ clean_text('eyecatcher_id') }}                              as eyecatcher_id,
        {{ clean_text('excerpt') }}                                    as excerpt,
        {{ clean_text('lede') }}                                       as lede,
        try_cast(impressions as bigint)                                as impressions,
        try_cast(clicks as bigint)                                     as clicks,
        try_cast(significance as double)                               as significance,
        try_cast(first_place as boolean)                               as first_place,
        try_cast(winner as boolean)                                    as winner,
        {{ archive_risk_flag() }}                                      as archive_randomization_risk,
        source_part,
        source_file
    from source

)

select
    package_id,
    source_row,
    clickability_test_id,
    created_at,
    updated_at,
    test_week,
    headline,
    eyecatcher_id,
    excerpt,
    lede,
    impressions,
    clicks,
    case when impressions > 0 then clicks::double / impressions end   as ctr,
    significance,
    first_place,
    winner,
    archive_randomization_risk,
    coalesce(
        created_at::date between date '{{ var("randomization_risk_start") }}'
                             and date '{{ var("randomization_risk_end") }}',
        false
    )                                                                  as in_risk_window,
    archive_randomization_risk or coalesce(in_risk_window, false)      as randomization_risk,
    {{ "'" ~ risk_col ~ "'" if risk_col else 'null' }}::varchar         as randomization_risk_source_column,
    source_part,
    source_file
from typed
