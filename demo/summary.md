# NYC Green Taxi Trips EDA Summary

* [Dataset Summary](#dataset-summary)
* [Unit of Observation](#unit-of-observation)
* [Columns Summary](#columns-summary)
* [Data Types Summary](#data-types-summary)
* [Missing Data](#missing-data)
* [Univariate Summaries](#univariate-summaries)
    * [VendorID](#VendorID)
    * [lpep_pickup_datetime](#lpep_pickup_datetime)
    * [lpep_dropoff_datetime](#lpep_dropoff_datetime)
    * [store_and_fwd_flag](#store_and_fwd_flag)
    * [RatecodeID](#RatecodeID)
    * [PULocationID](#PULocationID)
    * [DOLocationID](#DOLocationID)
    * [passenger_count](#passenger_count)
    * [trip_distance](#trip_distance)
    * [fare_amount](#fare_amount)
    * [extra](#extra)
    * [mta_tax](#mta_tax)
    * [tip_amount](#tip_amount)
    * [tolls_amount](#tolls_amount)
    * [improvement_surcharge](#improvement_surcharge)
    * [total_amount](#total_amount)
    * [payment_type](#payment_type)
    * [trip_type](#trip_type)
    * [congestion_surcharge](#congestion_surcharge)
    * [id](#id)

# Dataset Summary



|   Number of Rows |   Number of Unique Rows |   Number of Columns | Memory Usage   |
|-----------------:|------------------------:|--------------------:|:---------------|
|            63109 |                   63109 |                  20 | 20.2 MB        |

# Unit of Observation



The following column combinations serve as unique identifiers for the data:

* id

# Columns Summary

| Column                | Intedact Data Type   | Pandas Data Type   | Memory Usage   |
|:----------------------|:---------------------|:-------------------|:---------------|
| VendorID              | discrete             | float64            | 505.0 KB       |
| lpep_pickup_datetime  | discrete             | object             | 4.8 MB         |
| lpep_dropoff_datetime | discrete             | object             | 4.8 MB         |
| store_and_fwd_flag    | discrete             | object             | 3.0 MB         |
| RatecodeID            | discrete             | float64            | 505.0 KB       |
| PULocationID          | continuous           | int64              | 505.0 KB       |
| DOLocationID          | continuous           | int64              | 505.0 KB       |
| passenger_count       | discrete             | float64            | 505.0 KB       |
| trip_distance         | continuous           | float64            | 505.0 KB       |
| fare_amount           | continuous           | float64            | 505.0 KB       |
| extra                 | discrete             | float64            | 505.0 KB       |
| mta_tax               | discrete             | float64            | 505.0 KB       |
| tip_amount            | continuous           | float64            | 505.0 KB       |
| tolls_amount          | discrete             | float64            | 505.0 KB       |
| improvement_surcharge | discrete             | float64            | 505.0 KB       |
| total_amount          | continuous           | float64            | 505.0 KB       |
| payment_type          | discrete             | float64            | 505.0 KB       |
| trip_type             | discrete             | float64            | 505.0 KB       |
| congestion_surcharge  | discrete             | float64            | 505.0 KB       |
| id                    | continuous           | int64              | 505.0 KB       |

# Data Types Summary


![alt text](./images/data_types.png)

# Missing Data

# Univariate Summaries

## VendorID

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            38431 |              2 |           24678 |           39.1038 |)


![alt text](./images/VendorID_raw.png)

## lpep_pickup_datetime

**Intedact Data Type**: discrete

**Pandas Data Type**: object




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            63109 |          54347 |               0 |                 0 |)


![alt text](./images/lpep_pickup_datetime_raw.png)

## lpep_dropoff_datetime

**Intedact Data Type**: discrete

**Pandas Data Type**: object




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            63109 |          54584 |               0 |                 0 |)


![alt text](./images/lpep_dropoff_datetime_raw.png)

## store_and_fwd_flag

**Intedact Data Type**: discrete

**Pandas Data Type**: object




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            38431 |              2 |           24678 |           39.1038 |)


![alt text](./images/store_and_fwd_flag_raw.png)

## RatecodeID

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            38431 |              6 |           24678 |           39.1038 |)


![alt text](./images/RatecodeID_raw.png)

## PULocationID

**Intedact Data Type**: continuous

**Pandas Data Type**: int64




|   count_observed |   count_unique |   count_missing |   percent_missing |   min |   25% |   50% |    mean |   75% |   max |     std |   iqr |
|-----------------:|---------------:|----------------:|------------------:|------:|------:|------:|--------:|------:|------:|--------:|------:|
|            63109 |            250 |               0 |                 0 |     3 |    52 |    75 | 107.797 |   166 |   265 | 70.1656 |   114 |)


![alt text](./images/PULocationID_raw.png)

## DOLocationID

**Intedact Data Type**: continuous

**Pandas Data Type**: int64




|   count_observed |   count_unique |   count_missing |   percent_missing |   min |   25% |   50% |    mean |   75% |   max |   std |   iqr |
|-----------------:|---------------:|----------------:|------------------:|------:|------:|------:|--------:|------:|------:|------:|------:|
|            63109 |            258 |               0 |                 0 |     1 |    61 |   117 | 126.781 |   190 |   265 | 76.24 |   129 |)


![alt text](./images/DOLocationID_raw.png)

## passenger_count

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            38431 |             10 |           24678 |           39.1038 |)


![alt text](./images/passenger_count_raw.png)

## trip_distance

**Intedact Data Type**: continuous

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |   min |   25% |   50% |   mean |   75% |    max |     std |   iqr |
|-----------------:|---------------:|----------------:|------------------:|------:|------:|------:|-------:|------:|-------:|--------:|------:|
|            63109 |           2955 |               0 |                 0 |     0 |  1.25 |  2.66 | 45.843 |  6.34 | 162292 | 2184.13 |  5.09 |)


![alt text](./images/trip_distance_raw.png)

## fare_amount

**Intedact Data Type**: continuous

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |   min |   25% |   50% |    mean |   75% |   max |     std |   iqr |
|-----------------:|---------------:|----------------:|------------------:|------:|------:|------:|--------:|------:|------:|--------:|------:|
|            63109 |           4822 |               0 |                 0 |  -200 |     8 |  14.2 | 19.5774 | 25.28 | 335.5 | 16.3256 | 17.28 |)


![alt text](./images/fare_amount_raw.png)

## extra

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            63109 |             12 |               0 |                 0 |)


![alt text](./images/extra_raw.png)

## mta_tax

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            63109 |              4 |               0 |                 0 |)


![alt text](./images/mta_tax_raw.png)

## tip_amount

**Intedact Data Type**: continuous

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |   min |   25% |   50% |    mean |   75% |   max |     std |   iqr |
|-----------------:|---------------:|----------------:|------------------:|------:|------:|------:|--------:|------:|------:|--------:|------:|
|            63109 |            912 |               0 |                 0 | -0.99 |     0 |  1.56 | 1.58011 |  2.75 |   480 | 2.64507 |  2.75 |)


![alt text](./images/tip_amount_raw.png)

## tolls_amount

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            63109 |             46 |               0 |                 0 |)


![alt text](./images/tolls_amount_raw.png)

## improvement_surcharge

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            63109 |              3 |               0 |                 0 |)


![alt text](./images/improvement_surcharge_raw.png)

## total_amount

**Intedact Data Type**: continuous

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |    min |   25% |   50% |    mean |   75% |   max |     std |   iqr |
|-----------------:|---------------:|----------------:|------------------:|-------:|------:|------:|--------:|------:|------:|--------:|------:|
|            63109 |           5452 |               0 |                 0 | -200.3 |  10.3 | 17.55 | 22.9887 | 29.24 | 498.8 | 17.9916 | 18.94 |)


![alt text](./images/total_amount_raw.png)

## payment_type

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            38431 |              5 |           24678 |           39.1038 |)


![alt text](./images/payment_type_raw.png)

## trip_type

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            38431 |              2 |           24678 |           39.1038 |)


![alt text](./images/trip_type_raw.png)

## congestion_surcharge

**Intedact Data Type**: discrete

**Pandas Data Type**: float64




|   count_observed |   count_unique |   count_missing |   percent_missing |
|-----------------:|---------------:|----------------:|------------------:|
|            38431 |              3 |           24678 |           39.1038 |)


![alt text](./images/congestion_surcharge_raw.png)

## id

**Intedact Data Type**: continuous

**Pandas Data Type**: int64




|   count_observed |   count_unique |   count_missing |   percent_missing |   min |   25% |   50% |   mean |   75% |   max |     std |   iqr |
|-----------------:|---------------:|----------------:|------------------:|------:|------:|------:|-------:|------:|------:|--------:|------:|
|            63109 |          63109 |               0 |                 0 |     0 | 15777 | 31554 |  31554 | 47331 | 63108 | 18218.1 | 31554 |)


![alt text](./images/id_raw.png)BOOM!!