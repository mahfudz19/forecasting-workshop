import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class StockDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for text in texts:
            # Tokenize setiap "kalimat" harga (string tunggal)
            encodings = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.input_ids.append(encodings["input_ids"])
            self.attn_masks.append(encodings["attention_mask"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx].flatten(),
            "attention_mask": self.attn_masks[idx].flatten(),
        }


# Kelas Model LSTM Sederhana
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Core:
    def __init__(self, window_size: int = 10, epochs: int = 100):
        self.window_size = window_size
        self.epochs = epochs
        pass

    def main_feature(self):
        return "This is the main feature of the application."

    def helper_function(self, value):
        return value * 2

    def create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i : (i + self.window_size)])
            y.append(data[i + self.window_size])
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
            return data[["Close"]]  # Fokus pada harga penutupan untuk time series
        except Exception as e:
            raise ValueError(f"Gagal mengambil data untuk {symbol}: {str(e)}")

    # Fungsi baru: Preprocessing data saham
    def preprocess_stock_data(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Preprocessing data saham untuk forecasting.
        - Bersihkan missing values.
        - Normalisasi dengan MinMaxScaler.
        - Split jadi train dan test.
        Mengembalikan: (data_train_scaled, data_test_scaled, scaler)
        """
        # 1. Bersihkan: Isi missing values dengan forward fill (isi dengan nilai sebelumnya)
        data = data.ffill()

        # 2. Normalisasi: Ubah skala Close ke 0-1
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[["Close"]])

        # 3. Split: Bagi data (80% train, 20% test)
        train_data, test_data = train_test_split(
            data_scaled, test_size=test_size, shuffle=False
        )

        return train_data, test_data, scaler

    def set_data(self, train_data: np.ndarray, test_data: np.ndarray):
        X_train, y_train = self.create_sequences(train_data)
        X_test, y_test = self.create_sequences(test_data)

        # Ubah data ke format tensor PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

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
        pbar = tqdm(range(self.epochs), desc="Training LSTM")

        # Gunakan tqdm untuk progress bar yang profesional
        for epoch in pbar:
            model.train()
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            # Update postfix pada objek pbar
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print("--- Robot LSTM Selesai Belajar ---")

    def forecast_with_lstm(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        scaler: MinMaxScaler,
    ) -> list:
        """
        Forecasting harga saham dengan LSTM, metode yang lebih cocok untuk time series numerik.
        """

        # --- Langkah 1: Siapkan Data untuk LSTM ---
        X_train, y_train, X_test, y_test = self.set_data(train_data, test_data)

        # --- Langkah 2: Buat dan Latih Model LSTM ---
        model = SimpleLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        self.start_train(model, X_train, y_train, optimizer, criterion)

        model.eval()

        with torch.no_grad():
            test_predict = model(X_test)

        # Ubah hasil prediksi kembali ke format numpy
        predictions = test_predict.numpy()

        # Inverse scale (ubah angka 0-1 kembali ke harga asli)
        predictions_scaled = scaler.inverse_transform(predictions).flatten().tolist()

        print(f"\nPrediksi harga LSTM selesai ({len(predictions_scaled)})")

        return predictions_scaled
