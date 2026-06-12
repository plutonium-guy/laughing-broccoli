# Databricks notebook source
# MAGIC %md
# MAGIC # Country Code -> BUKRS Lookup
# MAGIC
# MAGIC Input: ISO 2-letter country code (e.g. 'FR', 'AT', 'DE')
# MAGIC Output: distinct BUKRS (company codes) found in BSAD for that country
# MAGIC
# MAGIC Logic: BUKRS in SAP typically starts with country code prefix
# MAGIC (FR01, FR05, AT01, DE01, etc.) — uses first 2 chars to filter.

# COMMAND ----------

from pyspark.sql import functions as F

BSAD_TABLE = "hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsad"


# =========================================================
# OPTION 1 — Python function (parameterized)
# =========================================================
def get_bukrs_for_country(country_code):
    """
    Return distinct BUKRS values from BSAD where company code starts
    with the given 2-letter country prefix. Includes row count per BUKRS.
    """
    country_code = country_code.upper().strip()

    return (
        spark.table(BSAD_TABLE)
        .filter(
            F.upper(F.substring(F.col("BUKRS"), 1, 2)) == country_code
        )
        .groupBy("BUKRS")
        .agg(
            F.count("*").alias("row_count"),
            F.countDistinct("KUNNR").alias("distinct_customers"),
            F.min(F.to_date("zfbdt", "yyyyMMdd")).alias("earliest_baseline_date"),
            F.max(F.to_date("zfbdt", "yyyyMMdd")).alias("latest_baseline_date"),
        )
        .orderBy(F.col("row_count").desc())
    )


# =========================================================
# OPTION 2 — Multi-country bulk version
# =========================================================
def get_bukrs_for_countries(country_codes):
    """Same as above but accepts a list. Returns one DataFrame with country col."""
    countries_upper = [c.upper().strip() for c in country_codes]

    return (
        spark.table(BSAD_TABLE)
        .withColumn("country_code", F.upper(F.substring(F.col("BUKRS"), 1, 2)))
        .filter(F.col("country_code").isin(countries_upper))
        .groupBy("country_code", "BUKRS")
        .agg(
            F.count("*").alias("row_count"),
            F.countDistinct("KUNNR").alias("distinct_customers"),
        )
        .orderBy("country_code", F.col("row_count").desc())
    )


# =========================================================
# RUN
# =========================================================

# Single country
fr_bukrs = get_bukrs_for_country("FR")
display(fr_bukrs)

# Multiple countries
all_bukrs = get_bukrs_for_countries(["FR", "AT", "DE", "IT"])
display(all_bukrs)


# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL version (Databricks SQL cell)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Replace 'FR' with desired country code
# MAGIC SELECT
# MAGIC     BUKRS,
# MAGIC     COUNT(*)               AS row_count,
# MAGIC     COUNT(DISTINCT KUNNR)  AS distinct_customers,
# MAGIC     MIN(to_date(zfbdt, 'yyyyMMdd')) AS earliest_baseline_date,
# MAGIC     MAX(to_date(zfbdt, 'yyyyMMdd')) AS latest_baseline_date
# MAGIC FROM hive_metastore.t_erp_ibp_customerservice_rbp_conf.bsad
# MAGIC WHERE UPPER(SUBSTRING(BUKRS, 1, 2)) = 'FR'
# MAGIC GROUP BY BUKRS
# MAGIC ORDER BY row_count DESC;
