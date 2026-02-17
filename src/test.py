import json
import os
from os import makedirs

import pandas as pd
import torch
import torchmetrics

from src import figures
from src.models.losses import UserwiseAUCROC


def get_testcase_rankingmetrics(test_case: pd.DataFrame):
    # Input: model probabilities and targets of a test case
    # Output: percentile of this test case and raw position of the author's image

    sorted_ranking = test_case.sort_values("pred", ascending=False).reset_index(
        drop=True
    )

    dev_position = sorted_ranking["is_dev"].idxmax()

    return pd.DataFrame(
        {
            "dev_position": [dev_position],
            "percentile": [dev_position / len(sorted_ranking)],
            "id_user": sorted_ranking["id_user"][0],
            "id_restaurant": sorted_ranking["id_restaurant"][0],
        }
    )


def test_tripadvisor_authorship_task(datamodule, model_preds, args):
    makedirs("docs/" + datamodule.city, exist_ok=True)

    # Data for the percentile figures
    percentile_figure_data = {"city": datamodule.city, "metrics": []}
    recall_figure_data = {"city": datamodule.city, "metrics": []}
    ndcg_figure_data = {"city": datamodule.city, "metrics": []}

    # Collect summary metrics per model (added for experiment logging)
    results = {}

    for model in model_preds:
        print("=" * 50)
        print(model)
        print("=" * 50)

        # Copy to avoid in-place modification of original test dataframe
        test_set = datamodule.test_dataset.dataframe.copy()
        test_set["pred"] = model_preds[model]

        # Sanity check: predictions must match test size
        assert len(model_preds[model]) == len(test_set), (
            model,
            len(model_preds[model]),
            len(test_set),
        )

        # Basic prediction distribution check (DeepSets debugging)
        print(
            "[SANITY] pred mean/std:",
            float(test_set["pred"].mean()),
            float(test_set["pred"].std()),
        )

        train_set = datamodule.train_dataset.dataframe

        # Get no. of images in each test case
        images_per_testcase = (
            test_set.groupby("id_test")
            .size()
            .reset_index(name="testcase_num_images")
        )

        # Compute number of photos in each user's train set
        train_photos_per_user = (
            train_set[train_set["take"] == 1]
            .drop_duplicates(keep="first")
            .groupby("id_user")
            .size()
            .reset_index(name="author_num_train_photos")
        )

        # Compute the percentile metric of each test case
        test_cases = (
            test_set.groupby("id_test")
            .apply(get_testcase_rankingmetrics)
            .reset_index()
        )

        # Add user-level and test-case-level information
        test_cases = pd.merge(
            test_cases,
            train_photos_per_user,
            left_on="id_user",
            right_on="id_user",
            how="inner",
        )
        test_cases = pd.merge(
            test_cases,
            images_per_testcase,
            left_on="id_test",
            right_on="id_test",
            how="inner",
        )

        # Initialize figure data
        model_percentile_metrics = {
            "min_photos": [],
            "num_test_cases": [],
            "median_percentile": [],
            "model_name": model,
        }

        preds = torch.tensor(test_set["pred"], dtype=torch.float)
        target = torch.tensor(test_set["is_dev"], dtype=torch.long)
        indexes = torch.tensor(test_set["id_test"], dtype=torch.long)

        model_userwise_auroc = UserwiseAUCROC()(
            indexes=indexes, target=target, preds=preds
        )

        print("")
        print(f"AUC (all users, all test cases): {model_userwise_auroc:.3f}")
        print("")

        auc_all = float(model_userwise_auroc)

        # We only take into account restaurants with >10 photos
        test_cases = test_cases[test_cases["testcase_num_images"] >= 10]

        print(f"Min. imgs  Percentile  Test Cases")
        for i in range(1, 101):
            percentiles = test_cases[
                test_cases["author_num_train_photos"] >= i
            ]["percentile"]

            model_percentile_metrics["min_photos"].append(i)
            model_percentile_metrics["num_test_cases"].append(len(percentiles))
            model_percentile_metrics["median_percentile"].append(
                percentiles.median()
            )

        percentile_figure_data["metrics"].append(model_percentile_metrics)

        # For recall metric, only include users with >=10 train images
        test_cases = test_cases[test_cases["author_num_train_photos"] >= 10]

        model_recall_metrics = {"k": [], "Recall@10": [], "model_name": model}
        model_ndcg_metrics = {"k": [], "NDCG@10": [], "model_name": model}

        test_set = test_set[
            test_set["id_test"].isin(test_cases["id_test"])
        ].reset_index(drop=True)

        preds = torch.tensor(test_set["pred"], dtype=torch.float)
        target = torch.tensor(test_set["is_dev"], dtype=torch.long)
        indexes = torch.tensor(test_set["id_test"], dtype=torch.long)

        print("k  Recall@10  NDCG@10")

        recall_at_10 = None
        ndcg_at_10 = None

        for k in range(1, 11):
            recall_k = torchmetrics.RetrievalRecall(k=k)(
                preds=preds, target=target, indexes=indexes
            )
            ndcg_k = torchmetrics.RetrievalNormalizedDCG(k=k)(
                preds=preds, target=target, indexes=indexes
            )

            model_recall_metrics["k"].append(k)
            model_recall_metrics["Recall@10"].append(recall_k)
            model_ndcg_metrics["k"].append(k)
            model_ndcg_metrics["NDCG@10"].append(ndcg_k)

            if k == 10:
                recall_at_10 = float(recall_k)
                ndcg_at_10 = float(ndcg_k)

            print(f"{k:<3}{recall_k:<8.3f}{ndcg_k:.3f}")

        recall_figure_data["metrics"].append(model_recall_metrics)
        ndcg_figure_data["metrics"].append(model_ndcg_metrics)

        model_userwise_auroc = UserwiseAUCROC()(
            indexes=indexes, target=target, preds=preds
        )

        print("")
        print(
            f"AUC (users with >=10 photos, test cases with size >=10): {model_userwise_auroc:.3f}"
        )
        print("")

        auc_10plus = float(model_userwise_auroc)

        # Store summary metrics for external logging (DeepSets experiments)
        results[model] = {
            "auc_all": auc_all,
            "auc_10plus": auc_10plus,
            "recall@10_10plus": recall_at_10,
            "ndcg@10_10plus": ndcg_at_10,
        }

    figures.percentile_figure(percentile_figure_data)

    return results
