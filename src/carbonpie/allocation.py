import pandas as pd
import numpy as np
import itertools

Gg_to_Gt = 1e-6
C_to_CO2 = 44.01 / 12.01


def gf(
    dataframe,
    regions=["CAN"],
    time=[
        2010
    ],  # FIXME: must be a list or list-like, otherwise it throws an error on ['year'].isin() in L36 and 44
    remaining_budget=[1600],
    meta=["source", "ssp", "forcing", "model"],
):
    """Allocates a regional carbon budget using Grandfathering

    Based on the Electronic supplementary material (ESM1) from Van Den Berg et al.(2020) (https://doi.org/10.1007/s10584-019-02368-y).

    Args:
        dataframe (DataFrame): A DataFrame in long form containing the data     from the Gütschow et al database for target regions and global (world).
        regions (list, optional): List of acronyms (str) for each target region in the dataset. Defaults to ['CAN'].
        world (list, optional): Acronym for global. Defaults to ['EARTH'].
        time (list, optional): List of years (int) to be used to calculate grandfathering emissions. Defaults to 2010.
        remaining_budget (list, optional): List of remaining carbon budgets (int). Defaults to 1600.
        meta (list, optional): List of metadata columns used as index. Defaults to ['source', 'ssp', 'forcing', 'model'].


    Returns:
        _type_: _description_
    """
    try:
        country_data = dataframe[
            (dataframe["country"].isin(regions))
            & (dataframe["entity"] == "CO2")
            & (dataframe["year"].isin(time))
        ].copy()

        global_data = dataframe[
            (dataframe["country"] == "EARTH")
            & (dataframe["entity"] == "CO2")
            & (dataframe["year"].isin(time))
        ].copy()

        # https://stackoverflow.com/questions/37840043/pandas-unstack-column-values-into-new-columns
        e = country_data.pivot(
            index=meta, columns=["country", "year"], values="value"
        )  # (meta, country, year)

        E = global_data.pivot(
            index=meta, columns=["country", "year"], values="value"
        )  # (meta, country, year)

        B = np.array(remaining_budget)  # (budget)

        try:
            carbon_budgets = np.einsum(
                "mry,mry,b->mrby",
                e.to_numpy().reshape(len(e.index.to_list()), len(regions), len(time)),
                1 / E.to_numpy().reshape(len(E.index.to_list()), 1, len(time)),
                B,
            )

        except ValueError as err:
            # if there is only one region, without the 'reshape' this threw a value error because 127,3 != 127,1,3. this might be avoided with np squeeze: np.squeeze(e.to_numpy().reshape(len(e.index.to_list()),len(regions),len(years))).shape
            print(f"Encountered an error: {err}")

        # FIXME: The methods should return melted (tidy) dataframes, and leave the visualisation for later. they should go into a results table.

        idx = e.stack("country").index

        idx_tuples = [
            (idx_row + (budget_year,))
            for budget_year in remaining_budget
            for idx_row in idx
        ]

        new_idx, _ = pd.MultiIndex.from_tuples(
            idx_tuples, names=meta + ["country", "budget"]
        ).sortlevel()

        result = pd.DataFrame(
            data=carbon_budgets.reshape(-1, len(time)), index=new_idx, columns=time
        )

        result.columns = pd.MultiIndex.from_product([["gf"], time])

    except AttributeError as err:
        print(f"Encountered error {err}")

    return result


def iepc(
    dataframe,
    regions=["CAN"],
    start_year=[2010],
    end_year=[2100],
    remaining_budget=[1600],
    meta=["source", "ssp", "forcing", "model"],
):
    """_summary_

    Args:
        dataframe (DataFrame): A DataFrame in long form containing the data     from the Gütschow et al database for target regions and global (world).
        world_pop (DataFrame, optional): DataFrame containing global population; it is currently not included in the Gütschow dataset, and was thus calculated by summing over all countries in the dataset. Defaults to pop_df.
        regions (list, optional): _description_. Defaults to ["CAN"].
        world (list, optional): _description_. Defaults to ["EARTH"].
        start_year (list, optional): _description_. Defaults to [2010].
        end_year (list, optional): _description_. Defaults to [2100].
        remaining_budget (list, optional): _description_. Defaults to [1600].
        meta (list, optional): _description_. Defaults to ["source", "ssp", "forcing", "model"].

    Returns:
        _type_: _description_
    """
    try:
        # Retrieve relevant data from input dataframe
        country_data = dataframe[
            (dataframe["country"].isin(regions))
            & (dataframe["entity"] == "POP")
            & (dataframe["year"].isin(range(min(start_year), max(end_year) + 1)))
        ].copy()

        global_data = dataframe[
            (dataframe["country"] == "EARTH")
            & (dataframe["entity"] == "POP")
            & (dataframe["year"].isin(range(min(start_year), max(end_year) + 1)))
        ].copy()

        # FIXME: this may not be necessary?
        idx_len = len(
            country_data.set_index(meta + ["year"])
            .groupby(level=meta)  # [0, 1, 2, 3]
            .count()
            .index
        )  # previously, len(r_pop.index)

        idx = (
            country_data.pivot(
                index=meta + ["year"], columns=["country"], values="value"
            )
            .groupby(level=meta)  # [0, 1, 2, 3]
            .sum()
            .stack("country")
            .index
        )  # previously, r_pop.stack('country').index

        idx_tuples = [
            (idx_row + (budget_year,))
            for budget_year in remaining_budget
            for idx_row in idx
        ]

        new_idx, _ = pd.MultiIndex.from_tuples(
            idx_tuples, names=meta + ["country", "budget"]
        ).sortlevel()

        # Calculations
        periods = list(itertools.product(start_year, end_year))
        cols = ["_".join(map(str, x)) for x in periods]
        iepc_dict = {}

        for i, period in enumerate(periods):
            # FIXME: It might be more efficient to create r_pop and g_pop first with all the years, initialize results df, and THEN use an indexslice to calculate the p / P values
            r_pop = (
                country_data.loc[
                    country_data["year"].isin(range(period[0], period[1] + 1)), :
                ]
                .pivot(index=meta + ["year"], columns=["country"], values="value")
                .groupby(level=meta)  # [0, 1, 2, 3]
                .sum()
            )

            g_pop = (
                global_data.loc[
                    global_data["year"].isin(range(period[0], period[1] + 1)), :
                ]
                .pivot(index=meta + ["year"], columns=["country"], values="value")
                .groupby(level=meta)  # [0, 1, 2, 3]
                .sum()
            )

            p = r_pop.to_numpy().reshape(idx_len, -1)  # dimensions meta, country

            P = g_pop.to_numpy().reshape(idx_len, 1)  # dimensions meta, country

            B = np.array(remaining_budget)  # (budget)

            carbon_budgets = np.einsum(
                "mr,mr,b->mrb",
                p,  # pop by region
                1 / P,  # POP, world pop
                B,
            )  # returns (127, regions, budget)

            # https://stackoverflow.com/questions/71577514/valueerror-per-column-arrays-must-each-be-1-dimensional-when-trying-to-create-a
            iepc_dict[cols[i]] = carbon_budgets.reshape(-1, len(r_pop.columns)).tolist()

            res = pd.DataFrame(carbon_budgets.reshape(-1, 1))
            res.index = new_idx
            res.columns = pd.MultiIndex.from_product([["iepc"], iepc_dict.keys()])

    except AttributeError as err:
        print(f"Encountered error {err}")

    return res


def pcc(
    gf_df=None,
    iepc_df=None,
    dataframe=pd.DataFrame(),  # budget_df
    regions=["CAN"],
    time=[2010, 2020],
    end_year=[2100],
    remaining_budget=[400, 1000, 1600],
    meta=["source", "ssp", "forcing", "model"],
    weighting_factor=0.5,
):
    # FIXME this seems very inefficient; why not just use the dataframes I already calculated earlier (pass resulting dataframes)?
    if gf_df is None:
        gf_df = gf(
            dataframe=dataframe,
            regions=regions,
            time=time,
            remaining_budget=remaining_budget,
            meta=meta,
        )

    if iepc_df is None:
        iepc_df = iepc(
            dataframe=dataframe,
            regions=regions,
            start_year=time,
            end_year=end_year,
            remaining_budget=remaining_budget,
            meta=meta,
        )

    # Adjust the gf results based on how many end_year there are, because gf produces one column per start_year (time), whereas iepc returns the combinations of start_year and end_year
    gf_values = np.repeat(gf_df.to_numpy(), len(end_year), axis=1)
    pcc_values = (gf_values * (1 - weighting_factor)) + (iepc_df * weighting_factor)

    result = pd.DataFrame(pcc_values, index=iepc_df.index)

    result.columns = pd.MultiIndex.from_product(
        [["pcc"], iepc_df.columns.get_level_values(1).to_list()]
    )

    return result


def ecpc(
    dataframe,
    regions=["CAN"],
    historical_start=[1850, 1970, 1990],
    time=[2010],
    end_year=[2100],
    remaining_budget=[
        400,
        1000,
        1600,
    ],
    meta=["source", "ssp", "forcing", "model"],
    discount_rate=[0.02],
):
    # Retrieve relevant data from input dataframe
    country_data = dataframe[
        (dataframe["country"].isin(regions))
        & (dataframe["entity"].isin(["CO2", "POP"]))
        & (dataframe["year"].isin(range(min(historical_start), max(end_year) + 1)))
    ].copy()

    global_data = dataframe[
        (dataframe["country"] == "EARTH")
        & (dataframe["entity"].isin(["CO2", "POP"]))
        & (dataframe["year"].isin(range(min(historical_start), max(end_year) + 1)))
    ].copy()

    # Retrieve relevant years
    periods = list(itertools.product(historical_start, end_year))
    cols = ["_".join(map(str, x)) for x in periods]
    ecpc_dict = {}

    for i, period in enumerate(periods):
        idx = pd.IndexSlice
        idx_subset = [
            "ssp",
            "forcing",
            "model",
            "country",
            "year",
        ]

        r_data = country_data.pivot(index=idx_subset, columns="entity", values="value")
        w_data = global_data.pivot(index=idx_subset, columns="entity", values="value")

        # Calculate debt
        nScenarios = len(set(r_data.index.droplevel(["country", "year"]).to_list()))
        nRegions = len(r_data.index.get_level_values("country").unique())
        nYears = len(
            set(
                r_data.loc[
                    idx[:, :, :, :, period[0] : time[0]], :
                ].index.get_level_values("year")
            )
        )

        p = (
            r_data.loc[idx[:, :, :, :, period[0] : time[0]], "POP"]
            .to_numpy()
            .reshape(
                nScenarios,  # scenarios
                nRegions,  # regions
                nYears,  # years
            )
        )  # m, r, y

        P = (
            w_data.loc[idx[:, :, :, :, period[0] : time[0]], "POP"]
            .to_numpy()
            .reshape(
                nScenarios,  # scenarios
                1,  # regions
                nYears,  # years
            )
        )  # m, r, y

        e = (
            r_data.loc[idx[:, :, :, :, period[0] : time[0]], "CO2"]
            .to_numpy()
            .reshape(
                nScenarios,  # scenarios
                nRegions,  # regions
                nYears,  # years
            )
        )  # m, r, y

        E = (
            w_data.loc[idx[:, :, :, :, period[0] : time[0]], "CO2"]
            .to_numpy()
            .reshape(
                nScenarios,  # scenarios
                1,  # regions
                nYears,  # years
            )
        )  # m, r, y

        dt = np.broadcast_to(
            discount_factor(discount_rate[0], period[0], time[0]),
            (nScenarios, 1, nYears),
        )  # FIXME Change to be able to apply different discount factors at once? the discount rate breaks if its passed a float instead of a list - it raises TypeError.

        debt = (p / P) * E * dt - (e * dt)
        debt = np.sum(debt, axis=2)
        debt = debt * Gg_to_Gt  # data already in co2

        # Calculate carbon budget
        f_p = (
            r_data.loc[idx[:, :, :, :, time[0] : period[1]], "POP"]
            .to_numpy()
            .reshape(
                len(
                    set(r_data.index.droplevel(["country", "year"]).to_list())
                ),  # scenarios
                len(r_data.index.get_level_values("country").unique()),  # regions
                len(
                    set(
                        r_data.loc[
                            idx[:, :, :, :, time[0] : period[1]], "POP"
                        ].index.get_level_values("year")
                    )
                ),  # years
            )
        )  # m, r, y

        f_P = (
            w_data.loc[idx[:, :, :, :, time[0] : period[1]], "POP"]
            .to_numpy()
            .reshape(
                len(
                    set(w_data.index.droplevel(["country", "year"]).to_list())
                ),  # scenarios
                len(w_data.index.get_level_values("country").unique()),  # regions
                len(
                    set(
                        w_data.loc[
                            idx[:, :, :, :, time[0] : period[1]], "POP"
                        ].index.get_level_values("year")
                    )
                ),  # years
            )
        )  # m, r, y

        # result = (np.sum(f_p, axis=2) / np.sum(f_P, axis=2)) * 400 + debt
        carbon_budget = np.einsum(
            "mr,mr,b->mrb",
            np.sum(f_p, axis=2),
            1 / np.sum(f_P, axis=2),
            np.array(remaining_budget),
        ) + np.repeat(debt, len(remaining_budget), axis=1).reshape(
            nScenarios, nRegions, len(remaining_budget)
        )

        ecpc_dict[cols[i]] = carbon_budget.reshape(
            -1
        ).tolist()  # FIXME may break if many regions at once?

    # Create results dataframe
    idx = r_data.groupby(by=["ssp", "forcing", "model", "country"]).count().index
    idx_tuples = [
        (idx_row + (budget_year,))
        for budget_year in remaining_budget
        for idx_row in idx
    ]

    new_idx, _ = pd.MultiIndex.from_tuples(
        idx_tuples, names=["ssp", "forcing", "model", "country", "budget"]
    ).sortlevel()

    result = pd.DataFrame(ecpc_dict, index=new_idx)
    result.columns = pd.MultiIndex.from_product([["ecpc"], ecpc_dict.keys()])

    # FIXME: this should be refactored, the function is so hard to read. Moreover, the prettify methods to render a good looking dataframe are not DRY, they are very similar to the ones in pcc. They should just become a single function to output the final dataset.

    return result


def select_distinct(dataframe, columns):
    """Select distinct existing column combinations from a dataframe"""
    result = set()
    for combination in zip(*[dataframe[col] for col in columns]):
        result.update((combination,))

    return result


def gdr():
    return None


def discount_factor(discount_rate, start_year, end_year):
    """Calculate a discount factor vector based on discount rate

    example:
        x = range(1850,2010+1)
        y = discount_factor(0.02, 1850, 2010)
        plt.plot(x,y)

    Args:
        discount_rate (_type_): _description_
        start_year (_type_): _description_
        end_year (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.array(
        [
            1 / (1 + discount_rate) ** (end_year - x)
            for x in list(range(start_year, end_year + 1))
        ]
    )


# TODO: unittests in pytest, e.g., check that weighting factor = 0 is equal to gf, and weighting factor =1 is equal to iepc

# FIXME the first filters on 'country data' and 'global data' are not DRY; also, they may be unnecessary as the data can be filtered directly through the initial sql query. Check whether it's returning the same df or not, or just gettint it through read_sql...
