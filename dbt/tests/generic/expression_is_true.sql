{#- Fails on every row where the expression is false (rows where it is null pass). -#}
{% test expression_is_true(model, expression, where_clause=none) %}
select *
from {{ model }}
where not ({{ expression }})
{% if where_clause %}and ({{ where_clause }}){% endif %}
{% endtest %}
