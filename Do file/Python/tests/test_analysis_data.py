"""Behavior checks for shared MICS data preparation."""

import importlib
import unittest
from unittest.mock import patch

import pandas as pd

try:
    analysis_data = importlib.import_module("_analysis_data")
except ModuleNotFoundError as error:
    if error.name != "_analysis_data":
        raise
    analysis_data = None


class SharedAnalysisDataTests(unittest.TestCase):
    def helper(self, name):
        function = getattr(analysis_data, name, None)
        self.assertTrue(callable(function), f"shared helper {name} is missing")
        return function

    def test_controls_for_household_and_child_samples(self):
        common = ("wealth", "urban")
        child = ("age", "male")

        self.assertEqual(
            self.helper("controls_for_sample")(common, child, False),
            ["wealth", "urban"],
        )
        self.assertEqual(
            self.helper("controls_for_sample")(common, child, True),
            ["wealth", "urban", "age", "male"],
        )

    def test_complete_case_rows_and_optional_metadata(self):
        data = pd.DataFrame({
            "y": [1.0, 2.0, None, 4.0],
            "d": [0, 1, 0, 1],
            "x": [5.0, 6.0, 7.0, None],
            "country_cat": [1, 1, 1, 1],
            "Cluster_var": [10, 10, 11, 12],
            "extra": [100.0, None, None, 400.0],
        })

        sample = self.helper("complete_case_sample")(
            data,
            "y",
            "d",
            ["x"],
            extra_columns=("extra",),
        )

        self.assertEqual(sample["y"].tolist(), [1.0, 2.0])
        self.assertEqual(sample["d"].tolist(), [0, 1])
        self.assertEqual(sample["x"].tolist(), [5.0, 6.0])
        self.assertEqual(sample["extra"].iloc[0], 100.0)
        self.assertTrue(pd.isna(sample["extra"].iloc[1]))
        self.assertEqual(sample.index.tolist(), [0, 1])

    def test_make_frame_preserves_dummy_and_cluster_columns(self):
        data = pd.DataFrame({
            "y": [10.0, 20.0, 30.0],
            "d": [0, 1, 0],
            "income": [1.0, 2.0, 3.0],
            "water_source": ["tap", "well", "tap"],
            "country_cat": [1, 2, 1],
            "Cluster_var": [5, 5, 9],
        })

        frame, x_columns = self.helper("make_frame")(
            data,
            "y",
            "d",
            ["income", "water_source"],
            categorical_controls=("water_source", "country_cat"),
        )

        self.assertEqual(
            x_columns,
            ["income", "water_source_well", "country_cat_2", "_cluster_model_code"],
        )
        self.assertEqual(frame["water_source_well"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(frame["country_cat_2"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(frame["_cluster_model_code"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(frame["Cluster_var"].tolist(), [5, 5, 9])
        self.assertNotIn("Cluster_var", x_columns)

    def test_load_data_filters_country_before_reproducible_sample(self):
        source = pd.DataFrame({
            "id": range(40),
            "y": [float(i) for i in range(40)],
            "x": [float(i) for i in range(40)],
            "country_cat": [1] * 20 + [2] * 20,
            "water_treatment": [i % 2 for i in range(40)],
            "WQ15_g": [i % 4 for i in range(40)],
            "Cluster_var": [i // 2 for i in range(40)],
            "RiskSource": [i % 3 for i in range(40)],
        })
        expected_columns = sorted([
            "x",
            "y",
            "water_treatment",
            "WQ15_g",
            "country_cat",
            "Cluster_var",
            "RiskSource",
        ])
        selected_source = source.loc[:, expected_columns]
        expected = selected_source.loc[
            selected_source["country_cat"].eq(2)
        ].sample(frac=0.25, random_state=17).reset_index(drop=True)

        load_data = self.helper("load_analysis_data")
        def read_selected_columns(path, *, columns, convert_categoricals):
            return source.loc[:, columns].copy()

        with patch.object(
            analysis_data.pd,
            "read_stata",
            side_effect=read_selected_columns,
        ) as read_stata:
            result = load_data(
                "fake.dta",
                "y",
                ["x"],
                country_codes=(2,),
                quick_sample=True,
                sample_fraction=0.25,
                sample_seed=17,
            )

        pd.testing.assert_frame_equal(result, expected)
        self.assertTrue(result["country_cat"].eq(2).all())
        self.assertEqual(
            read_stata.call_args.kwargs["columns"],
            expected_columns,
        )
        self.assertFalse(read_stata.call_args.kwargs["convert_categoricals"])


if __name__ == "__main__":
    unittest.main()
