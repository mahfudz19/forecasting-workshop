import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Core:
    def __init__(self):
        pass

    def main_feature(self):
        return "This is the main feature of the application."

    def helper_function(self, value):
        return value * 2

    # Fungsi baru: Ambil data saham historis
    def fetch_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Mengambil data harga saham dari Yahoo Finance.
        - symbol: Kode saham, e.g., 'AAPL' untuk Apple.
        - period: Periode data, e.g., '1y' untuk 1 tahun.
        Mengembalikan DataFrame dengan kolom seperti 'Close' (harga penutupan).
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data[["Close"]]  # Fokus pada harga penutupan untuk time series
        except Exception as e:
            raise ValueError(f"Gagal mengambil data untuk {symbol}: {str(e)}")

    # Fungsi baru: Preprocessing data saham
    def preprocess_stock_data(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> tuple:
        """
        Preprocessing data saham untuk forecasting.
        - Bersihkan missing values.
        - Normalisasi dengan MinMaxScaler.
        - Split jadi train dan test.
        Mengembalikan: (data_train_scaled, data_test_scaled, scaler)
        """
        # 1. Bersihkan: Isi missing values dengan forward fill (isi dengan nilai sebelumnya)
        data = data.fillna(method="ffill")

        # 2. Normalisasi: Ubah skala Close ke 0-1
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[["Close"]])

        # 3. Split: Bagi data (80% train, 20% test)
        train_data, test_data = train_test_split(
            data_scaled, test_size=test_size, shuffle=False
        )

        return train_data, test_data, scaler
