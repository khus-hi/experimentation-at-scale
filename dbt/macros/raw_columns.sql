{#- Column names of raw.packages, lower-cased. Empty at parse time. -#}
{% macro raw_column_names() %}
    {%- if not execute -%}{{ return([]) }}{%- endif -%}
    {%- set cols = adapter.get_columns_in_relation(source('upworthy', 'packages')) -%}
    {{ return(cols | map(attribute='name') | map('lower') | list) }}
{% endmacro %}

{#- The archive's own randomization-risk column, if the loaded files carry one (June 2024
    update). Returns the first column whose name contains one of var('randomization_risk_hints'),
    or none. -#}
{% macro archive_risk_column() %}
    {%- for c in raw_column_names() -%}
        {%- for hint in var('randomization_risk_hints') -%}
            {%- if hint in c -%}{{ return(c) }}{%- endif -%}
        {%- endfor -%}
    {%- endfor -%}
    {{ return(none) }}
{% endmacro %}

{#- SQL boolean for the archive's risk flag: true for true/1/yes-like values, false when the
    column is missing. -#}
{% macro archive_risk_flag() %}
    {%- set c = archive_risk_column() -%}
    {%- if c is none -%}
        false
    {%- else -%}
        coalesce(
            try_cast({{ adapter.quote(c) }} as boolean),
            lower(trim(cast({{ adapter.quote(c) }} as varchar)))
                not in ('', '0', 'false', 'f', 'no', 'n', 'none', 'na', 'nan', 'null'),
            false
        )
    {%- endif -%}
{% endmacro %}

{#- Text cleanup: trim, collapse empty strings to null. -#}
{% macro clean_text(col) -%}
    nullif(trim(cast({{ col }} as varchar)), '')
{%- endmacro %}
