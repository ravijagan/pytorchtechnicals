
multiplesymbols = """
with esx as
(
select * , avg(close) over minwindow  as esxclose, date_trunc('minute', t_date) as minx from  
  "ES_val" oe 
where date_trunc('day' , t_date) > '2023-09-01'
WINDOW minwindow as ( partition by date_trunc('minute', t_date)  order by date_trunc('minute', t_date))
),
spxx as 
(
select * , avg(close) over minwindow  as spxxclose, date_trunc('minute', t_date) as minx from  "SPX_val" oe 
where date_trunc('day' , t_date) > '2023-09-01'
WINDOW minwindow as ( partition by date_trunc('minute', t_date)  order by date_trunc('minute', t_date))
) ,
tltxx as 
(
select * , avg(close) over minwindow  as tltxxclose, date_trunc('minute', t_date) as minx from  "TLT_val" oe 
where date_trunc('day' , t_date) > '2023-09-01'
WINDOW minwindow as ( partition by date_trunc('minute', t_date)  order by date_trunc('minute', t_date))
) ,
sofrx as 
(
select * , avg(close) over minwindow  as sofrxclose, date_trunc('minute', t_date) as minx from  "SOFR3_val" oe 
where date_trunc('day' , t_date) > '2023-09-01'
WINDOW minwindow as ( partition by date_trunc('minute', t_date)  order by date_trunc('minute', t_date))
)

-- daily diff
	 --select date_trunc('day' , minx) , avg(esx.close) as esc, avg(spxx.close) as spc, 
	 --min (esx.close - spxx.close) as mindiff,
	 --avg (esx.close - spxx.close) as diff,
	-- max (esx.close - spxx.close) as maxdiff
 	 --from esx join spxx using(minx) group by 1 order by 1
 -- end of daily diff 


select esx.minx as minx, to_char(esx.minx, 'YYYYMMDDHH24mm')::bigint as minute_to_int,  avg(esxclose) as es , 
           avg(spxxclose)  as spx  , 
           avg(tltxxclose) as tltx, avg(sofrxclose) as sofr
--esx.close - spxx.close -30 as diff,
 --esx.close - spxx.close  - avg(esx.close - spxx.close) over (ORDER By minx ROWS BETWEEN 32 PRECEDING AND CURRENT ROW)  as feat,
 --lead(spxx.close,1) over (order by minx, spxx.close)  - spxx.close as gain1,
--lead(spxx.close,2) over (order by minx, spxx.close)  - spxx.close as gain2,
--lead(spxx.close,5) over (order by minx, spxx.close)  - spxx.close as gain5

from esx left join spxx using(minx) 
left join tltxx using(minx)
left join sofrx  using(minx)
group by 1,2 
order by esx.minx """

q1min2022 = """with aux as(
select date_trunc( 'minute' , (t_date + interval '3 hours') )as minutex, * from spxmin -- ibkr_spx
),
elt as (
select distinct
    stock_symbol, -- underlying_symbol,
    minutex as t_date,
    first_value(close) over minwindow  as open, 
    max(close) over minwindow  as high,
    min(close) over minwindow as  low,
    last_value(close) over minwindow  as close,
    trade_volume, 
    vwap, 
    bid,
    ask 
    --,*
from aux 
--group by 1,2
WINDOW minwindow as ( partition by minutex order by minutex)
)
--select * from elt order by t_date
select 
                --t_date, 
    to_char(t_date, 'YYYYMMDDHH24MI')::float as t_date, -- may be stripped out later
                EXTRACT(MONTH FROM t_date)  as month,
                 EXTRACT(day FROM t_date )  as day ,
                EXTRACT(DOW FROM t_date )  as dow  ,
                   (EXTRACT(HOUR FROM t_date)*100) + EXTRACT(Minute FROM t_date)  as hour_min, 
                   high , low, bid, ask, open , close ,
                   AVG(close) OVER(ORDER BY t_date ROWS BETWEEN  5 following and 8 FOLLOWING  ) - close AS five_toeight_min_delta_future
     from elt
            where
                close > 2000 --data cleansing 
                and EXTRACT(HOUR FROM t_date)  >= 9  
                and EXTRACT(HOUR FROM t_date)  <= 16  
                order by t_date  ; 
     
"""
