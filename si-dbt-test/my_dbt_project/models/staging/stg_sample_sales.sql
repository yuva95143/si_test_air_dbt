-- Simple staging model for testing
{{ config(materialized='view') }}

select
    id,
    customer_name,
    product,
    quantity,
    price,
    order_date,
    quantity * price as total_amount
from {{ ref('sample_sales') }}
