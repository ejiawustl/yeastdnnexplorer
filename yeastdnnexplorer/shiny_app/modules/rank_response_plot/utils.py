import logging

from scipy.stats import binom

from yeastdnnexplorer.interface import ExpressionAPI, PromoterSetSigAPI, RankResponseAPI


# Function to calculate confidence interval for the random value
def binom_ci(trials, random_prob, alpha=0.05):
    lower_bound = binom.ppf(alpha / 2, trials, random_prob) / trials
    upper_bound = binom.ppf(1 - alpha / 2, trials, random_prob) / trials
    return lower_bound, upper_bound


async def submit_rank_response_job(
    pss: PromoterSetSigAPI,
    expression: ExpressionAPI,
    rr: RankResponseAPI,
    logger: logging.Logger,
):
    logger.info("Retrieving expression data")
    expression_metadata = await expression.read()

    logger.info("Retrieved expression metadata")
    pss_metadata = await pss.read()

    expression_metadata_df = expression_metadata.get("metadata")

    # rename the `id` column to expression_id
    expression_metadata_df.rename(columns={"id": "expression_id"}, inplace=True)

    pss_metadata_df = pss_metadata.get("metadata")

    # rename the `id` column to promotersetsig_id
    pss_metadata_df.rename(columns={"id": "promotersetsig_id"}, inplace=True)

    # filter the expression_metadata_df based on the unique set of values in `regulator`
    # in pss_metadata_df. There may be multiple rows in expression_metadata_df
    # with the same value in `regulator` column. That is OK.
    regulator_values = pss_metadata_df["regulator_symbol"].unique()
    expression_metadata_df_subset = expression_metadata_df[
        expression_metadata_df["regulator_symbol"].isin(regulator_values)
    ]

    # select the columns regulator, and expression_id from expression_metadata_df_subset
    # and regulator, promotersetsig_id from promotersetsig_metadata_df. Left join
    # expression_metadata_df_subset with pss_metadata_df on the `regulator` column.
    # This will result in a dataframe with columns regulator, expression_id,
    # promotersetsig_id
    merged_df = expression_metadata_df_subset.loc[
        :, ["regulator_symbol", "expression_id"]
    ].merge(
        pss_metadata_df.loc[:, ["regulator_symbol", "promotersetsig_id"]],
        on="regulator_symbol",
        how="left",
    )

    # iterate over the rows in merged_df and for each row, call set the rr.params and
    # call rr.read()
    post_data = []
    for index, row in merged_df.iterrows():

        post_data.append(
            {
                "promotersetsig_ids": [row["promotersetsig_id"]],
                "expression_ids": [row["expression_id"]],
                "rank_by_binding_effect": "true",
            }
        )

    group_id = await rr.submit(post_dict=post_data)  # type: ignore

    # rr_res = await rr.retrieve(group_id)

    # rr_metadata = rr_res.get("metadata")

    # rr_df = rr_res.get("data").get(rr_res.get("metadata").id[0])

    # output_dict.update({key: (rr_metadata, rr_df)})

    return group_id
