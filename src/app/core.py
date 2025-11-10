import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict


# Kelas Model LSTM Sederhana
class SimpleLSTM(nn.Module):
    def __init__(
        self, input_size=1, hidden_size=50, num_layers=1, output_size=1, dropout=0.0
    ):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Core:
    def __init__(
        self,
        window_size: int = 10,
        epochs: int = 100,
        hidden_size: int = 50,
        num_layers: int = 1,
        lr: float = 0.001,
        batch_size: int = 32,
        dropout: float = 0.0,
    ):
        self.window_size = window_size
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout

    def main_feature(self):
        return "This is the main feature of the application."

    def helper_function(self, value):
        return value * 2

    def create_sequences(self, data: np.ndarray, close_price_index: int):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i : (i + self.window_size)])
            y.append(data[i + self.window_size, close_price_index])
        return np.array(X), np.array(y)

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
            return data[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            raise ValueError(f"Gagal mengambil data untuk {symbol}: {str(e)}")

    # Fungsi baru: Preprocessing data saham
    def preprocess_stock_data(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, MinMaxScaler], list]:
        """
        Preprocessing data saham untuk forecasting.
        - Menskalakan setiap fitur secara independen.
        - Split data mentah terlebih dahulu untuk mencegah data leakage.
        - Fit scaler HANYA pada data training.
        - Transform kedua set data.
        """
        # 1. Tambahkan indikator dan siapkan fitur
        data = data.ffill()
        data.ta.sma(length=20, append=True)
        data.ta.rsi(length=14, append=True)
        data.ta.obv(append=True)

        features = ["Open", "High", "Low", "Close", "Volume", "SMA_20", "RSI_14", "OBV"]
        # features = ["Open", "High", "Low", "Close", "Volume"]

        data_featured = data[features].dropna()

        # 2. Split data MENTAH terlebih dahulu
        train_df, test_df = train_test_split(
            data_featured, test_size=test_size, shuffle=False
        )

        # 3. Buat dan FIT scaler untuk SETIAP FITUR secara terpisah
        scalers = {}
        train_data_scaled = train_df.copy()
        test_data_scaled = test_df.copy()

        for feature_name in features:
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Fit scaler HANYA pada data training untuk kolom saat ini
            scaler.fit(train_df[[feature_name]])
            scalers[feature_name] = scaler  # Simpan scaler di dictionary

            # Transform kolom fitur yang sesuai di data training dan test
            train_data_scaled[feature_name] = scaler.transform(train_df[[feature_name]])
            test_data_scaled[feature_name] = scaler.transform(test_df[[feature_name]])

        return (
            train_data_scaled.values,  # Kirim sebagai numpy array
            test_data_scaled.values,  # Kirim sebagai numpy array
            scalers,
            features,
        )

    def set_data(
        self, train_data: np.ndarray, test_data: np.ndarray, close_price_index: int
    ):
        X_train, y_train = self.create_sequences(train_data, close_price_index)
        X_test, y_test = self.create_sequences(test_data, close_price_index)

        # Ubah data ke format tensor PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        return X_train, y_train, X_test, y_test

    def start_train(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ):
        print("\n--- Robot LSTM Mulai Belajar ---")
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        pbar = tqdm(range(self.epochs), desc="Training LSTM")
        for epoch in pbar:
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                outputs = model(xb)
                optimizer.zero_grad()
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            pbar.set_postfix(loss=f"{avg:.4f}")
        print("--- Robot LSTM Selesai Belajar ---")

    def forecast_with_lstm(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        scalers: Dict[str, MinMaxScaler],
        close_price_index: int,
    ) -> list:
        """
        Forecasting harga saham dengan LSTM, metode yang lebih cocok untuk time series numerik.
        """

        # --- Langkah 1: Siapkan Data untuk LSTM ---
        X_train, y_train, X_test, y_test = self.set_data(
            train_data, test_data, close_price_index
        )
        num_features = X_train.shape[2]

        # --- Langkah 2: Buat dan Latih Model LSTM ---
        model = SimpleLSTM(
            input_size=num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.start_train(model, X_train, y_train, optimizer, criterion)

        model.eval()

        with torch.no_grad():
            test_predict = model(X_test)

        # Ubah hasil prediksi kembali ke format numpy
        predictions = test_predict.numpy()

        # Dapatkan scaler khusus untuk kolom 'Close'
        close_scaler = scalers["Close"]  # Ambil scaler 'Close' dari dictionary

        # Lakukan inverse transform pada prediksi
        predictions_original_scale = close_scaler.inverse_transform(
            predictions
        ).flatten()

        print(f"Prediksi harga LSTM selesai ({len(predictions_original_scale)})")

        return predictions_original_scale.tolist()
