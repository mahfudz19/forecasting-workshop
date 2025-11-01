import unittest
from src.app.core import Core  # Perbaiki import: gunakan Core, bukan YourMainClass
import pandas as pd


class TestCore(unittest.TestCase):

    def setUp(self):
        self.instance = Core()

    def test_main_feature(self):
        result = self.instance.main_feature()
        self.assertEqual(result, "This is the main feature of the application.")

    def test_helper_function(self):
        result = self.instance.helper_function(5)
        self.assertEqual(result, 10)

    # Test baru untuk fetch_stock_data
    def test_fetch_stock_data(self):
        data = self.instance.fetch_stock_data("AAPL", "1mo")  # Test dengan data 1 bulan
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn("Close", data.columns)
        self.assertGreater(len(data), 0)  # Pastikan ada data


if __name__ == "__main__":
    unittest.main()
