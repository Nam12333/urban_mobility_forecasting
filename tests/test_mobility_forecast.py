import unittest

from src.mobility_forecast import (
    build_matrix,
    generate_synthetic_records,
    metrics,
    run_pipeline,
    train_test_split,
)


class MobilityForecastTests(unittest.TestCase):
    def test_synthetic_data_has_expected_shape(self):
        records = generate_synthetic_records(rows=48)
        self.assertEqual(len(records), 48)
        self.assertTrue(all(record.count > 0 for record in records))

    def test_feature_matrix_matches_records(self):
        records = generate_synthetic_records(rows=72)
        feature_names, matrix, target = build_matrix(records)
        self.assertEqual(len(matrix), len(records))
        self.assertEqual(len(target), len(records))
        self.assertIn("commute_hour", feature_names)
        self.assertIn("weather_bad", feature_names)

    def test_train_test_split_preserves_all_rows(self):
        records = generate_synthetic_records(rows=100)
        _, matrix, target = build_matrix(records)
        x_train, x_test, y_train, y_test = train_test_split(matrix, target)
        self.assertEqual(len(x_train) + len(x_test), 100)
        self.assertEqual(len(y_train) + len(y_test), 100)

    def test_metrics_return_expected_keys(self):
        result = metrics([10, 20, 30], [12, 18, 33])
        self.assertEqual(set(result), {"mae", "rmse", "r2"})
        self.assertGreater(result["r2"], 0)

    def test_pipeline_beats_mean_baseline_on_sample_data(self):
        report = run_pipeline(download=False)
        self.assertIn("ridge_regression", report)
        self.assertLess(report["ridge_regression"]["rmse"], report["baseline"]["rmse"])
        self.assertGreater(report["ridge_regression"]["r2"], 0.55)


if __name__ == "__main__":
    unittest.main()

