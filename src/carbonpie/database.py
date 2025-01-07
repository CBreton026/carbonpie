#!/usr/bin/env python3

""" Database operations

This file provides a context manager and initialization function for the database.

When run as a script, it will check if the database exists and create it if it does not.

First test of full implementation took ~13m40s and pulled ~12 GB RAM (5 more than base consumption on my machine).
"""

import os
import toml
import glob
import pandas as pd
import sqlite3
import traceback
import subprocess
from urllib.request import pathname2url
from contextlib import contextmanager


def initialize_database(
    database,
    data_path,
    countries,
    chunksize,
    un_groups,
    sql_files,
):  # FIXME should be passed a config file?
    print("Verifying that the database exists...")
    # https://stackoverflow.com/questions/12932607/how-to-check-if-a-sqlite3-database-exists-in-python
    try:
        db_uri = "file:{}?mode=rw".format(pathname2url(database))
        conn = sqlite3.connect(db_uri, uri=True)
        print("Database exists.")
        conn.close()

    except sqlite3.OperationalError:
        print("Database does not exist. Creating...")
        # https://stackoverflow.com/questions/2887878/importing-a-csv-file-into-a-sqlite3-database-table-using-python
        subprocess.run(
            [
                "sqlite3",
                str(database),
                "-cmd",
                ".mode csv",
                ".import " + str(data_path) + " data",
            ],  # FIXME:  "since subprocess passes all follow-ons to -cmd as quoted strings, you need to double up your backslashes if you have a windows directory path". Also, expects a comma separator; otherwise needs ".separator ;" command.
            capture_output=True,
        )
        print("Database created.")

    print("Validating that table 'data' is tidy...")
    try:
        with db_ops(database) as conn:
            query = "SELECT * FROM data LIMIT 1;"
            df = pd.read_sql(query, conn)

            if "year" not in df.columns:
                print("Table is not tidy. Converting...")
                query = "SELECT * FROM data;"
                result = pd.read_sql_query(query, conn, chunksize=chunksize)

                for i, df in enumerate(result):
                    print(f"Converting dataframe {i} to longform.")
                    # see discussion on chunksize: https://www.architecture-performance.fr/ap_blog/reading-a-sql-table-by-chunks-with-pandas/
                    df = dataframe_prep(df)
                    print(df.size)
                    df.to_sql(
                        "longform_data",
                        conn,
                        chunksize=chunksize,
                        method="multi",
                        index=False,
                        if_exists="append",
                    )
                print("Conversion complete.")

                print("Replacing table 'data' with longform data...")
                cur = conn.cursor()
                cur.executescript(
                    """          
                DROP TABLE data;
                ALTER TABLE "longform_data" RENAME TO "data";
                """
                )
                print("Table replaced. Cleaning up...")
                cur = conn.cursor()
                cur.execute("VACUUM")
                print("Database vacuumed.")

            else:
                print("Table is tidy. Skipping conversion.")

    except sqlite3.OperationalError as err:
        print(f"SqLite3 error: {err}.")
        traceback.print_exc()

    print("Validating that secondary tables exist...")
    try:
        with db_ops(database) as conn:
            # https://stackoverflow.com/questions/1601151/how-do-i-check-in-sqlite-whether-a-table-exists
            query = """
                SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type="table" AND name=?);
                """  # SELECT EXISTS returns 0 or 1 based on wether the SELECT 1 returns 1 or None.

            missing_tables = []
            for table_name in ["countries", "manual_sums", "scenarios"]:
                print(f"Checking for table {table_name}...")
                cur = conn.cursor()
                table_exists = cur.execute(query, (table_name,))
                table_exists = table_exists.fetchall()[
                    0
                ]  # Retrieve if it's true (1,) or false (0,) and save the value, it's emptied after the fetchall()
                if any(table_exists):
                    print(f"Table {table_name} already exists.")
                else:
                    print(f"Table {table_name} does not exist.")
                    missing_tables.append(table_name)

            if missing_tables:
                print("Creating missing tables...")
                if "scenarios" in missing_tables:
                    print("Creating table 'scenarios'...")
                    create_scenarios_table(database)

                if "countries" in missing_tables:
                    print("Creating table 'countries'...")
                    create_countries_table(database, countries)

                if "manual_sums" in missing_tables:
                    print("Creating table 'manual_sums'...")
                    create_manual_sums_table(database, un_groups, sql_files)

                print("Missing tables created.")
                print("Database initialization complete.")

            else:
                print("All tables exist. Skipping creation.")
                print("Database initialization complete.")

    except sqlite3.OperationalError as err:
        print(f"SqLite3 error: {err}.")
        traceback.print_exc()


@contextmanager
def db_ops(db_path):
    """_summary_
    # https://stackoverflow.com/questions/67436362/decorator-for-sqlite3/67436763#67436763

    Args:
        db_path (_type_): _description_
    """
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    except Exception as err:
        conn.rollback()
        raise err
    else:
        conn.commit()
    finally:
        conn.close()


def dataframe_prep(dataframe):
    """_summary_

    meant to be used with the raw dataframe imported from the Gütshow csv, accepts a dataframe, and returns it in long form. cf tidy data
    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe = dataframe.set_index(
        ["source", "scenario", "country", "category", "entity", "unit"]
    )
    dataframe.columns.name = "year"
    dataframe = dataframe.stack("year").reset_index()
    dataframe.rename({0: "value"}, axis=1, inplace=True)
    dataframe = dataframe.astype(
        {
            "source": "str",
            "scenario": "str",
            "country": "str",
            "category": "str",
            "entity": "str",
            "unit": "str",
            "year": "int",
            "value": "float",
        }
    )

    return dataframe


def create_scenarios_table(database):
    try:
        with db_ops(database) as conn:
            query = """          
                SELECT DISTINCT scenario FROM data;
                """
            scenarios = pd.read_sql_query(query, conn)

            # Create the scenarios dataframe
            # FIXME: This it not DRY
            scenarios["ssp"] = (
                pd.Series(scenarios["scenario"])
                .astype("str")
                .map(lambda x: str(x)[0:4])
            ).values
            scenarios["forcing"] = (
                pd.Series(scenarios["scenario"])
                .astype("str")
                .map(lambda x: str(x)[4:6])
            ).values
            scenarios["model"] = (
                pd.Series(scenarios["scenario"]).astype("str").map(lambda x: str(x)[6:])
            ).values

            scenarios.to_sql(
                "scenarios", conn, index=False, if_exists="fail"
            )  # 'append', 'replace'

    except sqlite3.OperationalError as err:
        print(f"SqLite3 error: {err}.")
        traceback.print_exc()


def create_countries_table(database, countries):
    try:
        with db_ops(database) as conn:
            # Retrieve the countries dataframe
            countries = pd.read_csv(countries, sep=",", encoding="utf-8")

            countries.to_sql(
                "countries", conn, index=False, if_exists="fail"
            )  # 'append', 'replace'

    except sqlite3.OperationalError as err:
        print(f"SqLite3 error: {err}.")
        traceback.print_exc()


def create_manual_sums_table(database, un_groups, sql_files):
    try:
        with db_ops(database) as conn:
            # Creating the manual_sums
            cur = conn.cursor()
            query = """CREATE TABLE IF NOT EXISTS manual_sums (
                source TEXT,
                scenario TEXT,
                country TEXT,
                category TEXT,
                entity TEXT,
                unit TEXT,
                year INTEGER,
                value REAL,
                count INTEGER
            );"""
            cur.execute(query)

            for infile in glob.glob(sql_files):
                # https://stackoverflow.com/questions/29522298/sqlite-insert-into-select-how-to-insert-data-of-join-of-3-existing-tables-i
                # https://www.somethingaboutdata.com/something-about-python/python-read-sql-data
                print(f"Working on file {infile}...")
                with open(infile, "r") as sql_file:
                    sql = sql_file.read()

                    # adding population sums
                    print("adding population sums")
                    cur = conn.cursor()
                    cur.execute(sql, ["POP"])

                    # adding GDPPPP sums
                    print("adding GDPPPP sums")
                    cur = conn.cursor()
                    cur.execute(sql, ["GDPPPP"])

    except sqlite3.OperationalError as err:
        print(f"SqLite3 error: {err}.")
        traceback.print_exc()


if __name__ == "__main__":
    CONFIG_PATH = os.path.abspath("./results/config/config.toml")

    try:
        print("Loading run parameters from config file.\n")
        with open(CONFIG_PATH, "r") as infile:
            config = toml.load(infile)

        COUNTRIES_PATH = config["paths"]["countries"]
        DATA_PATH = config["paths"]["data"]
        SQL_GLOB = config["paths"]["sql_glob"]

        UN_GROUPS = config["params"]["un_groups"]

        DATABASE_PATH = config["database"]["path"]
        CHUNKSIZE = config["database"]["chunksize"]

        initialize_database(
            database=DATABASE_PATH,
            data_path=DATA_PATH,
            countries=COUNTRIES_PATH,
            chunksize=CHUNKSIZE,
            un_groups=UN_GROUPS,
            sql_files=SQL_GLOB,
        )

    except Exception as err:
        print(f"Unexpected {type(err)}: {err}")
        traceback.print_exc()
