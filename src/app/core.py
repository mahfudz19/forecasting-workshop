import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    GPT2Model,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset
import numpy as np
import json


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
        data = data.ffill()

        # 2. Normalisasi: Ubah skala Close ke 0-1
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[["Close"]])

        # 3. Split: Bagi data (80% train, 20% test)
        train_data, test_data = train_test_split(
            data_scaled, test_size=test_size, shuffle=False
        )

        return train_data, test_data, scaler

    def forecast_with_gpt(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        scaler: MinMaxScaler,
        window_size: int = 10,
    ) -> list:
        """
        Forecasting harga saham dengan GPT-2 (diadaptasi untuk time series).
        - Convert data numerik ke token teks.
        - Fine-tune GPT untuk predict token berikutnya.
        - Inverse ke harga asli.
        """

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        def num_to_token(num: float) -> str:
            return f"price_{num:.4f}"

        def token_to_num(token: str) -> float:
            if "price_" in token:
                try:
                    return float(token.replace("price_", ""))
                except ValueError:
                    return 0.0
            return 0.0

        train_full_sequences_texts = []
        for i in range(len(train_data) - window_size):
            input_seq_tokens = [
                num_to_token(train_data[j][0]) for j in range(i, i + window_size)
            ]
            target_token = num_to_token(train_data[i + window_size][0])
            full_sequence_text = " ".join(input_seq_tokens + [target_token])
            train_full_sequences_texts.append(full_sequence_text)

        test_sequences_for_prediction = []
        for i in range(len(test_data) - window_size):
            input_seq_tokens = [
                num_to_token(test_data[j][0]) for j in range(i, i + window_size)
            ]
            target_token = num_to_token(test_data[i + window_size][0])
            test_sequences_for_prediction.append(
                {"input_seq": input_seq_tokens, "target_token": target_token}
            )

        train_dataset = StockDataset(train_full_sequences_texts, tokenizer)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        print("--- Robot GPT Selesai Belajar ---")

        # --- Minta Robot Menebak Harga Saham Masa Depan (Prediksi) ---

        predictions = []
        # Atur model ke mode evaluasi (tidak belajar lagi, hanya menebak)
        model.eval()

        print("\n--- Robot GPT Mulai Menebak Harga Saham ---")
        # Gunakan torch.no_grad() agar tidak menghitung gradient saat prediksi (lebih cepat dan hemat memori)
        with torch.no_grad():
            for i, seq_data in enumerate(test_sequences_for_prediction):
                input_seq_tokens = seq_data["input_seq"]

                # Gabungkan token input menjadi satu string untuk tokenizer
                input_text = " ".join(input_seq_tokens)

                # Tokenisasi input dan pindahkan ke device yang sama dengan model (CPU/MPS)
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

                # Minta model untuk generate (menebak) 1 token berikutnya
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,  # Hanya generate 1 token berikutnya
                    do_sample=False,  # Jangan pakai sampling, ambil token paling mungkin
                    pad_token_id=tokenizer.eos_token_id,  # Penting untuk generate
                )

                # Ambil token yang diprediksi (token terakhir dari output)
                predicted_token_id = outputs[0][-1]
                predicted_token = tokenizer.decode(
                    predicted_token_id, skip_special_tokens=True
                )

                # Ubah "kata-kata harga" yang diprediksi kembali menjadi angka (0-1)
                predicted_num = token_to_num(predicted_token)
                predictions.append(predicted_num)

                # Print progres prediksi setiap beberapa langkah
                if (i + 1) % 10 == 0 or (i + 1) == len(test_sequences_for_prediction):
                    print(
                        f"Prediksi ke-{i+1}/{len(test_sequences_for_prediction)}: Input '{input_text}' -> Tebakan '{predicted_num}'"
                    )

        print("--- Robot GPT Selesai Menebak ---")

        # Inverse scale (ubah angka 0-1 kembali ke harga asli)
        predictions_scaled = []
        if predictions:  # Pastikan ada prediksi sebelum inverse_transform
            predictions_scaled = (
                scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                .flatten()
                .tolist()
            )

        print(f"\nPrediksi harga (skala asli) 5 hari pertama: {predictions_scaled[:5]}")
        print(f"Total prediksi harga: {len(predictions_scaled)}")

        return predictions_scaled  # Kembalikan daftar prediksi harga asli
