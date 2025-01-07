import os
import traceback
import toml
import shutil
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from itertools import chain
from carbonpie import allocation as alloc
from carbonpie import database as db

# TODO: change print statements to proper logging

CONFIG_PATH = os.path.abspath("./results/config/config.toml")
with open(CONFIG_PATH, "r") as infile:
    config = toml.load(infile)
    FIG_START_YEAR = config["params"]["start_year"][0]
    FIG_BUDGET = config["params"]["remaining_budget"]
    SAVE_PATH = config["paths"]["fig_dir"]
    OUTDIR = config["paths"][
        "outdir"
    ]  # TODO: create a unique 'run' output dir in results, e.g., (https://stackoverflow.com/questions/67780603/how-to-create-a-unique-folder-name-location-path-in-windows); https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp


def main():
    print("Opening config file...")
    try:
        with open(CONFIG_PATH, "r") as infile:
            config = toml.load(infile)

        print("Loading paths from config file...")
        COUNTRIES_PATH = config["paths"]["countries"]
        DATA_PATH = config["paths"]["data"]
        SQL_GLOB = config["paths"]["sql_glob"]

        print("Loading run parameters from config file...")
        REMAINING_CARBON_BUDGET = config["params"]["remaining_budget"]
        HIST_START = config["params"]["historic_start_year"]
        START_YEAR = config["params"]["start_year"]
        END_YEAR = config["params"]["end_year"]
        COUNTRIES = config["params"]["target_countries"]
        UN_GROUPS = config["params"]["un_groups"]
        PCC_WEIGHTING_FACTOR = config["params"]["pcc_weighting_factor"]
        ECPC_DISCOUNT_RATE = config["params"]["ecpc_discount_rate"]

        print("Loading database parameters from config file...")
        DATABASE_PATH = config["database"]["path"]
        CHUNKSIZE = config["database"]["chunksize"]
        print("Configuration loaded successfully. \n")

        print("Saving a copy of the config file... \n")
        shutil.copy2(
            CONFIG_PATH, OUTDIR
        )  # FIXME ideally, update OUT_DIR to be run-specific

    except Exception as err:
        print(f"Unexpected {type(err)}: {err}")
        traceback.print_exc()

    print("Initializing the main database...")
    try:
        db.initialize_database(
            database=DATABASE_PATH,
            data_path=DATA_PATH,
            countries=COUNTRIES_PATH,
            chunksize=CHUNKSIZE,
            un_groups=UN_GROUPS,
            sql_files=SQL_GLOB,
        )
        print("Database initialized. \n")

    except sqlite3.OperationalError as err:
        print(f"Database error. {type(err)}: {err}")
        traceback.print_exc()

    # Retrieve full dataset for allocation methods
    print("Preparing dataset for allocation methods...")
    try:
        with db.db_ops(DATABASE_PATH) as conn:
            # Query adapted from https://stackoverflow.com/questions/1309989/parameter-substitution-for-a-sqlite-in-clause;
            query = """
            SELECT d.source, s.ssp, s.forcing, s.model, d.country, d.entity, d.unit, d.year, d.value
            FROM data as d
            LEFT JOIN scenarios as s
            ON d.scenario = s.scenario
            WHERE d.country IN (%s)
            AND d.entity IN ("POP", "GDPPPP", "CO2")
            AND d.unit IN ("Million2011GKD","ThousandPers", "Gg")
            UNION ALL
            SELECT m.source, s.ssp, s.forcing, s.model, m.country, m.entity, m.unit, m.year, m.value
            FROM manual_sums as m
            LEFT JOIN scenarios as s
            ON m.scenario = s.scenario
            WHERE m.country IN (%s)
            AND m.entity IN ("POP", "GDPPPP");""" % (
                ",".join("?" * len(COUNTRIES)),
                ",".join("?" * len(COUNTRIES)),
            )

            df = pd.read_sql_query(
                query, conn, params=tuple(chain(tuple(COUNTRIES), tuple(COUNTRIES)))
            )
            print("Data prepared. \n")

    except sqlite3.OperationalError as err:
        print(f"Database error. {type(err)}: {err}")
        traceback.print_exc()

    # print(df.sample(15))

    print("Attempting allocation methods...")
    try:
        print("Grandfathering...")
        gf_df = alloc.gf(
            dataframe=df,
            regions=COUNTRIES,
            time=START_YEAR,
            remaining_budget=REMAINING_CARBON_BUDGET,
            meta=["ssp", "forcing", "model"],
        )
        # print(gf_df.head())
        print("Grandfathering done. \n")

        print("IEPC...")
        iepc_df = alloc.iepc(
            dataframe=df,
            regions=COUNTRIES,
            start_year=START_YEAR,
            end_year=END_YEAR,
            remaining_budget=REMAINING_CARBON_BUDGET,
            meta=["ssp", "forcing", "model"],
        )
        # print(iepc_df.head())
        print("IEPC done. \n")

        print("PCC...")
        pcc_df = alloc.pcc(
            gf_df=gf_df,
            iepc_df=iepc_df,
            regions=COUNTRIES,
            time=START_YEAR,
            end_year=END_YEAR,
            remaining_budget=REMAINING_CARBON_BUDGET,
            weighting_factor=PCC_WEIGHTING_FACTOR,
            meta=["ssp", "forcing", "model"],
        )
        # print(pcc_df.head())
        print("PCC done. \n")

        print("ECPC...")
        ecpc_df = alloc.ecpc(
            df,
            regions=COUNTRIES,
            historical_start=HIST_START,
            time=START_YEAR,
            end_year=END_YEAR,
            remaining_budget=REMAINING_CARBON_BUDGET,
            meta=["ssp", "forcing", "model"],
            discount_rate=ECPC_DISCOUNT_RATE,
        )
        # print(ecpc_df.head())
        print("ECPC done. \n")

    except Exception as err:
        print(f"Unexpected {type(err)}: {err}")
        traceback.print_exc()

    # Create figures
    print("Creating figures...")
    alloc_df = {
        "gf": gf_df,
        "iepc": iepc_df,
        "pcc": pcc_df,
        "ecpc": ecpc_df,
    }
    compiled_df = pd.concat(alloc_df.values(), axis=1)
    compiled_df.columns.names = ["method", "variant"]
    # print(compiled_df)

    plot_df = (
        compiled_df.copy()
        .stack(["method", "variant"])
        .reset_index()
        .rename(columns={0: "value"})
        .astype(
            {
                "ssp": "category",
                "forcing": "category",
                "model": "category",
                "country": "category",
                "method": "category",
                "variant": "category",
            }
        )
    )
    # print(plot_df)

    print("Saving figure dataframe... \n")
    # This is mainly to save the dataset for exploratory analysis in a notebook, e.g., carbon_budget.ipynb
    # plot_df[plot_df["country"] == "CAN"].to_csv(
    #     OUTDIR + "test_results.csv", mode="w", index=False
    # )
    plot_df.to_csv(OUTDIR + "test_results.csv", mode="w", index=False)

    # Create a numerical DF and convert all categorical variables to numeric
    # https://www.statology.org/convert-categorical-variable-to-numeric-pandas/
    # identify all categorical variables
    num_df = plot_df[
        plot_df["country"] != "EARTH"
    ].copy()  # FILTER OUT 'EARTH' values; FIXME if more than one country, fig2 (which uses num_df) will have to be modified
    cat_columns = num_df.select_dtypes(["category"]).columns
    num_df[cat_columns] = num_df[cat_columns].apply(lambda x: pd.factorize(x)[0])

    # Extract the CO2 emissions for Canada for selected scenarios
    target_scenarios = [
        ("SSP1", "19"),
        ("SSP1", "26"),
        ("SSP2", "45"),
        ("SSP3", "70"),
        ("SSP4", "34"),
        ("SSP4", "60"),
        ("SSP5", "34"),
        ("SSP5", "85"),
    ]

    # Calculate budget by method and RCB
    fig5_data = (
        plot_df[plot_df["country"] == "CAN"]
        .groupby(by=["budget", "ssp", "forcing", "method"])
        .mean(numeric_only=True)
        .reset_index()
    )
    fig5_data = fig5_data.loc[
        fig5_data[["ssp", "forcing"]].apply(tuple, axis=1).isin(target_scenarios), :
    ]

    # Retrieve projected emissions for all target  scenarios (dataframe for figure 4)
    res = (
        df.loc[
            (df["country"] == "CAN")
            & (df["entity"] == "CO2")
            & (df["year"] >= FIG_START_YEAR)
            & (df[["ssp", "forcing"]].apply(tuple, axis=1).isin(target_scenarios)),
            ["ssp", "forcing", "year", "value"],
        ]
        .groupby(by=["ssp", "forcing", "year"])
        .agg({"value": ["mean", "std"]})
        .transform(lambda x: x * alloc.Gg_to_Gt)
    )

    fig4_data = res.groupby(by=["ssp", "forcing"]).cumsum().reset_index()
    fig4_data.drop(columns=("value", "std"), inplace=True)
    fig4_data.columns = ["ssp", "forcing", "year", "value"]

    # Calculate budget spent-by dates
    # for each budget-ssp-forcing combination, get a year for each budget (by method, the columns)
    budgets = fig5_data.pivot(
        index=["budget", "ssp", "forcing"], columns="method", values="value"
    ).reset_index()

    nyears = len(fig4_data)
    newcol = np.concatenate(
        (
            np.full((nyears, 1), REMAINING_CARBON_BUDGET[0]),
            np.full((nyears, 1), REMAINING_CARBON_BUDGET[1]),
            np.full((nyears, 1), REMAINING_CARBON_BUDGET[2]),
        )
    )  # FIXME REMAINING_CARBON_BUDGET will fail if less than 3 values
    fig4_data = pd.DataFrame(
        np.tile(fig4_data.to_numpy(), (3, 1)),
        columns=["ssp", "forcing", "year", "value"],
    )
    fig4_data["budget"] = newcol

    group_a = budgets.groupby(by=["budget", "ssp", "forcing"])
    group_b = fig4_data.groupby(by=["budget", "ssp", "forcing"])
    budget_spent = []
    for name, group in group_b:
        spent_by = np.argmax(
            np.greater_equal(
                np.divide(
                    group["value"].to_numpy().reshape(-1, 1),
                    group_a.get_group(name).loc[:, "ecpc":"pcc"].to_numpy(),
                ),
                1,
            ),
            axis=0,
        )  # argmax get first 'max', which is first true value from boolean sorting; first year in which the budget threshold is passed
        budget_spent.append(list(name) + list(spent_by))

    bsb = pd.DataFrame(budget_spent, columns=budgets.columns)
    fig6_data = (
        bsb.set_index(["budget", "ssp", "forcing"])
        .stack("method")
        .reset_index()
        .rename({0: "year"}, axis=1)
    )

    # fig6_data.to_csv(
    #     "fig6.csv"
    # )

    print("Processing figures...")
    try:
        figure1(df)
        figure2(num_df, plot_df)
        figure5(fig5_data)
        figure6(fig6_data)
        figure4(fig4_data)
        print("Figures saved.\n")

    except Exception as err:
        print(f"Unexpected {type(err)}: {err}")
        traceback.print_exc()
        print("Figures not saved.\n")

    except Exception as err:
        print(f"Unexpected {type(err)}: {err}")
        traceback.print_exc()


def figure1(dataframe):
    """_summary_

    plots population, co2 and gdp by country

    Args:
        dataframe (_type_): _description_
    """

    palette = sns.color_palette("colorblind")
    sns.set_theme(context="paper", palette=palette)

    hue = dataframe[["ssp", "forcing"]].apply(tuple, axis=1)

    g = sns.relplot(
        data=dataframe,
        x="year",
        y="value",
        col="country",
        row="entity",
        hue=hue,  # create 'scenario'? ssp+forcing
        height=8.27 / 3,
        aspect=16 / 9,
        legend=False,
        facet_kws={"sharex": True, "sharey": "row"},
    )

    g.savefig(
        SAVE_PATH + "figure1.png", dpi=300, bbox_inches="tight", transparent=False
    )  # https://stackoverflow.com/questions/39870642/how-to-plot-a-high-resolution-graph


def figure2(num_data, ref_data):
    # https://plotly.com/python/parallel-coordinates-plot/
    # https://plotly.com/python/reference/parcoords/
    BUDGET_MIN = -15
    BUDGET_MAX = 18

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=num_data["value"],
                colorscale="viridis",
                showscale=True,
                cmin=BUDGET_MIN,
                cmax=BUDGET_MAX,
            ),
            dimensions=list(
                [
                    dict(
                        range=[200, 1000],
                        tickvals=FIG_BUDGET,  # [247, 504, 944]
                        label="Remaining carbon budget",
                        values=num_data["budget"],
                    ),
                    dict(
                        range=[0, 4],
                        tickvals=[0, 1, 2, 3, 4],
                        ticktext=ref_data["ssp"].unique(),
                        label="SSP scenario",
                        values=num_data["ssp"],
                    ),
                    dict(
                        range=[0, 5],
                        tickvals=[0, 1, 2, 3, 4, 5],
                        ticktext=ref_data["forcing"].unique(),
                        label="Radiative forcing",
                        values=num_data["forcing"],
                    ),
                    dict(
                        range=[-1, 4],
                        tickvals=[0, 1, 2, 3],
                        ticktext=[x for x in list(ref_data["method"].unique())],
                        label="Allocation method",
                        values=num_data["method"],
                    ),
                    # dict(range = [0,5],
                    #     tickvals = [0,1,2,3,4,5],
                    #     ticktext= ref_data['model'].unique(),
                    #     label= 'Modèle',
                    #     values= num_data['model']),
                    dict(
                        range=[BUDGET_MIN, BUDGET_MAX],
                        label="Carbon budget",
                        values=num_data["value"],
                    ),
                ]
            ),
            unselected=dict(line=dict(color="lightgray", opacity=0.5)),
        )
    )

    # voir https://plotly.com/python/figure-labels/
    # fig.update_layout(
    #     title="Canada's carbon budget estimates",
    # )

    fig.write_image(
        file=SAVE_PATH + "newplot_small_alt.png",
        format="png",
        width=640,
        height=400,
        scale=3,
        engine="kaleido",
    )  # 1200 x 750
    # for better quality, consider using d3js although it seems complicated. (https://d3-graph-gallery.com/graph/parallel_basic.html)


def figure3(dataframe):
    palette = sns.color_palette("colorblind")
    sns.set_theme(context="paper", palette=palette)

    g = sns.catplot(
        data=dataframe[
            dataframe["budget"] == FIG_BUDGET[2]
        ],  # 944, FIXME check if this fails if fewer than 3 values
        x="method",
        y="value",
        hue="ssp",
        # col='method',
        # row='forcing',
        # hue='model',
        kind="box",
        sharex=True,
        # sharey='row'
    )

    g.savefig(
        SAVE_PATH + "figure3.png", dpi=300, bbox_inches="tight", transparent=False
    )  # https://stackoverflow.com/questions/39870642/how-to-plot-a-high-resolution-graph


def figure4(dataframe):
    palette = sns.color_palette("colorblind")
    sns.set_theme(context="paper", palette=palette)

    hue = dataframe[["ssp", "forcing"]].apply(tuple, axis=1)

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.lineplot(data=dataframe, x="year", y="value", hue=hue, ax=ax)
    budget = [-8.040005, 14.303867, 5.811804, 10.057835]  # FIXME harcoded
    ax.hlines(
        y=budget,
        xmin=FIG_START_YEAR,  # 2022
        xmax=2100,
        colors="purple",
        linestyles="--",
        lw=2,
    )
    ax.text(FIG_START_YEAR, budget[0], "ecpc ", ha="right", va="center")  # 2022
    ax.text(FIG_START_YEAR, budget[1], "gf ", ha="right", va="center")  # 2022
    ax.text(FIG_START_YEAR, budget[2], "iepc ", ha="right", va="center")  # 2022
    ax.text(FIG_START_YEAR, budget[3], "pcc ", ha="right", va="center")  # 2022

    plt.savefig(
        SAVE_PATH + "figure4.png", dpi=300, bbox_inches="tight", transparent=False
    )  # https://stackoverflow.com/questions/39870642/how-to-plot-a-high-resolution-graph


def figure5(dataframe):
    palette = sns.color_palette("colorblind")
    sns.set_theme(context="paper", palette=palette)

    hue = dataframe[["ssp", "forcing"]].apply(tuple, axis=1)

    g = sns.catplot(
        data=dataframe,
        x="method",
        y="value",
        hue=hue,
        col="budget",
    )

    g.savefig(
        SAVE_PATH + "figure5.png", dpi=300, bbox_inches="tight", transparent=False
    )  # https://stackoverflow.com/questions/39870642/how-to-plot-a-high-resolution-graph


def figure6(dataframe):
    palette = sns.color_palette("colorblind")
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=palette,
        #   font_scale=,
    )

    hue = dataframe[["ssp", "forcing"]].apply(tuple, axis=1)

    g = sns.catplot(
        data=dataframe,
        x="year",
        y="method",
        # row='budget',
        hue=hue,
        sharex=True,
        kind="strip",
        height=3,  # inches
        aspect=4 / 3,
    )

    g.set_axis_labels("Remaining years", "Allocation method")
    g.despine(left=True)

    # https://seaborn.pydata.org/generated/seaborn.move_legend.html
    sns.move_legend(
        g,
        "center left",
        bbox_to_anchor=(0.85, 0.5),
        ncol=1,  # len(hue)
        title=None,
        frameon=False,
    )

    g.savefig(
        SAVE_PATH + "figure6.png", dpi=300, bbox_inches="tight", transparent=False
    )  # https://stackoverflow.com/questions/39870642/how-to-plot-a-high-resolution-graph; https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html#matplotlib.figure.Figure.savefig


if __name__ == "__main__":
    main()
