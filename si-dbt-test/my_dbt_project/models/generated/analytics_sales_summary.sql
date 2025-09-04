-- Simple analytics model
{{ config(materialized='table') }}

select
    product,
    count(*) as order_count,
    sum(quantity) as total_quantity,
    sum(total_amount) as total_revenue,
    avg(total_amount) as avg_order_value
from {{ ref('stg_sample_sales') }}
group by product
