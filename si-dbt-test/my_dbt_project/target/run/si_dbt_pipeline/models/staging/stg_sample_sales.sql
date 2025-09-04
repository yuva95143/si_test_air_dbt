
    
    create view main."stg_sample_sales" as
    -- Simple staging model for testing


select
    id,
    customer_name,
    product,
    quantity,
    price,
    order_date,
    quantity * price as total_amount
from main."sample_sales";