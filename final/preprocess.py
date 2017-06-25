#!/usr/bin/env python3

import numpy as np
import pandas as pd

import sys
import math


def categorize(column, prefix, limit=50, add_other=True):
    categories = column.value_counts().index[:limit]

    result = pd.DataFrame([{
        "{}_{}".format(prefix, c): 1 if element == c else 0
        for c in categories
    } for element in column])

    if add_other:
        result["{}_my_other".format(prefix)] = result.apply(
            lambda r: 1 if (r == 0).all() else 0, axis=1)

    return result


def clean(data):
    cleaned_data = data.copy()

    na_values = [
        None,
        0,
        -2e-8,
        "",
        "None",
        "none",
        "Unknown",
        "unknown",
        "Not Known",
        "not known",
    ]

    cleaned_data = cleaned_data.replace(na_values, np.nan)

    # time

    date = cleaned_data["date_recorded"]
    year = date.map(lambda x: x.year)
    month = date.map(lambda x: x.month)
    weekday = date.map(lambda x: x.dayofweek)
    cleaned_data = cleaned_data.join(categorize(year, "year", add_other=False))
    cleaned_data = cleaned_data.join(
        categorize(month, "month", add_other=False))
    cleaned_data = cleaned_data.join(
        categorize(weekday, "weekday", add_other=False))

    construction_year = cleaned_data["construction_year"]
    construction_year_median = construction_year.median()
    construction_year_min = construction_year.min()
    cleaned_data["construction_year"] = construction_year.map(
        lambda x: construction_year_median if pd.isnull(x) else x)
    cleaned_data["age"] = year - cleaned_data["construction_year"]
    cleaned_data["construction_year"] = construction_year.map(
        lambda x: x - construction_year_min)

    # longitude & latitude

    longitude = cleaned_data.groupby("lga")["longitude"].mean().to_dict()
    latitude = cleaned_data.groupby("lga")["latitude"].mean().to_dict()
    longitude.update({"Geita": 32.2314})
    latitude.update({"Geita": -2.885})

    for k, v in cleaned_data.iterrows():
        if pd.isnull(cleaned_data.loc[k, "longitude"]):
            cleaned_data.loc[k, "longitude"] = longitude[v["lga"]]
        if pd.isnull(cleaned_data.loc[k, "latitude"]):
            cleaned_data.loc[k, "latitude"] = latitude[v["lga"]]

    # gps_height

    height = cleaned_data["gps_height"]
    height_median = height.median()
    height = height.fillna(height_median)
    height[height < 0] = height_median
    cleaned_data["height"] = height

    # region

    region = cleaned_data["region"]
    cleaned_data = cleaned_data.join(
        categorize(region, "region", add_other=False))

    # lga

    lga = cleaned_data["lga"]
    cleaned_data = cleaned_data.join(categorize(lga, "lga", add_other=False))

    # basin

    basin = cleaned_data["basin"]
    cleaned_data = cleaned_data.join(
        categorize(basin, "basin", add_other=False))

    # amount_tsh

    amount = cleaned_data["amount_tsh"]
    amount_median = amount.median()
    amount = amount.fillna(amount_median)
    amount = amount.map(lambda x: math.log1p(x))
    cleaned_data["amount"] = amount
    cleaned_data["amount_square"] = amount.map(lambda x: math.pow(x, 2))
    cleaned_data["amount_third_quantile"] = amount.map(
        lambda x: 1 if x > 3.044 else 0)

    # population

    population = cleaned_data["population"]
    population_median = population.median()
    population = population.fillna(population_median)
    population = population.map(lambda x: math.log1p(x))
    cleaned_data["population"] = population
    cleaned_data["population_below"] = population.map(
        lambda x: 1 if x < 2 else 0)

    # funder

    funder = cleaned_data["funder"]
    cleaned_data = cleaned_data.join(categorize(funder, "funder"))

    # installer

    installer = cleaned_data["installer"]
    cleaned_data = cleaned_data.join(categorize(installer, "installer"))

    # management

    management = cleaned_data["management"]
    cleaned_data = cleaned_data.join(categorize(management, "management"))

    # scheme_management

    scheme_management = cleaned_data["scheme_management"]
    cleaned_data = cleaned_data.join(
        categorize(scheme_management, "scheme_management"))

    # extraction_type

    extraction_type = cleaned_data["extraction_type"]
    cleaned_data = cleaned_data.join(
        categorize(extraction_type, "extraction_type"))

    # payment

    payment = cleaned_data["payment"]
    cleaned_data = cleaned_data.join(categorize(payment, "payment"))

    # water_quality

    quality = cleaned_data["water_quality"]
    cleaned_data = cleaned_data.join(categorize(quality, "quality"))

    # quantity

    quantity = cleaned_data["quantity"]
    cleaned_data = cleaned_data.join(categorize(quantity, "quantity"))

    # source

    source = cleaned_data["source"]
    cleaned_data = cleaned_data.join(categorize(source, "source"))

    # waterpoint_type

    t = cleaned_data["waterpoint_type"]
    cleaned_data = cleaned_data.join(categorize(t, "type"))

    cleaned_data = cleaned_data.drop(
        [
            "amount_tsh",
            "date_recorded",
            "funder",
            "gps_height",
            "installer",
            "wpt_name",
            "num_private",
            "basin",
            "subvillage",
            "region",
            "region_code",
            "district_code",
            "lga",
            "ward",
            "public_meeting",
            "recorded_by",
            "scheme_management",
            "scheme_name",
            "permit",
            "extraction_type",
            "extraction_type_group",
            "extraction_type_class",
            "management",
            "management_group",
            "payment",
            "payment_type",
            "water_quality",
            "quality_group",
            "quantity",
            "quantity_group",
            "source",
            "source_type",
            "source_class",
            "waterpoint_type",
            "waterpoint_type_group",
        ],
        axis=1)

    return cleaned_data


if __name__ == "__main__":
    x_train_file_name = sys.argv[1]
    y_train_file_name = sys.argv[2]
    x_test_file_name = sys.argv[3]

    x_train = pd.read_csv(x_train_file_name, parse_dates=["date_recorded"])
    y_train = pd.read_csv(y_train_file_name)
    x_test = pd.read_csv(x_test_file_name, parse_dates=["date_recorded"])

    cleaned_x_train_test = clean(x_train.append(x_test, ignore_index=True))
    num_training_data = y_train.shape[0]
    clened_x_train = cleaned_x_train_test[:num_training_data]
    clened_x_test = cleaned_x_train_test[num_training_data:]

    clened_x_train.to_csv("cleaned_x_train.csv")
    clened_x_test.to_csv("cleaned_x_test.csv")
    clened_x_train.to_pickle("cleaned_x_train.p")
    clened_x_test.to_pickle("cleaned_x_test.p")
