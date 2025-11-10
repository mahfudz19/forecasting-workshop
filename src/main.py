from app.core import Core
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Dict
import pandas_ta as ta


def evaluasi_RMSE_MAE(predictions: list, y_test_actual_original_scale: list):
    print("\n--- Evaluasi Akurasi Robot LSTM ---")
    rmse = np.sqrt(mean_squared_error(y_test_actual_original_scale, predictions))
    mae = mean_absolute_error(y_test_actual_original_scale, predictions)

    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")

    persistence_preds = y_test_actual_original_scale[:-1]
    actual_for_persistence = y_test_actual_original_scale[1:]

    if len(persistence_preds) > 0:
        rmse_persistence = np.sqrt(
            mean_squared_error(actual_for_persistence, persistence_preds)
        )
        mae_persistence = mean_absolute_error(actual_for_persistence, persistence_preds)
        print("\n--- Perbandingan dengan Baseline (Persistence Model) ---")
        print(f"RMSE (Persistence): {rmse_persistence:.4f}")
        print(f"MAE (Persistence): {mae_persistence:.4f}")

        if rmse < rmse_persistence:
            print(
                ">>> KESIMPULAN: Model LSTM lebih baik dari baseline sederhana. (Bagus!)"
            )
        else:
            print(
                ">>> KESIMPULAN: Model LSTM belum lebih baik dari baseline sederhana."
            )

    print("--- Evaluasi Selesai ---")


def plot_predictions(
    actual: list, predictions: list, title: str = "Perbandingan Harga Asli vs Prediksi"
):
    """
    Membuat grafik perbandingan antara harga asli dan prediksi.
    """
    print("\n--- Visualisasi Prediksi ---")
    plt.figure(figsize=(14, 7))  # Ukuran grafik lebih besar
    plt.plot(actual, label="Harga Asli (y_test)", color="blue", linewidth=2)
    plt.plot(
        predictions, label="Prediksi (LSTM)", color="red", linestyle="--", linewidth=2
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Waktu (Hari)", fontsize=12)
    plt.ylabel("Harga Saham", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()  # Menyesuaikan layout agar tidak ada yang terpotong
    plt.show()
    print("--- Visualisasi Selesai ---")


def tableGenerate(
    scalers: Dict[str, MinMaxScaler],
    features: list,
    X_test_raw_scaled: np.ndarray,
    predictions: list,
    y_test_acl_ori_scl: list,
    close_price_index: int,
):
    # --- Tampilkan Tabel Perbandingan ---
    print("\n--- Tabel Perbandingan Prediksi ---")

    num_samples, window_size, num_features = X_test_raw_scaled.shape
    X_test_original_scale = np.zeros_like(X_test_raw_scaled)
    for i, feature_name in enumerate(features):
        scaler = scalers[feature_name]
        column_scaled = X_test_raw_scaled[:, :, i].reshape(-1, 1)
        column_original = scaler.inverse_transform(column_scaled)
        X_test_original_scale[:, :, i] = column_original.reshape(
            num_samples, window_size
        )

    # Buat list untuk data tabel
    table_data = []
    for i in range(len(predictions)):
        # Ambil 3 nilai terakhir dari kolom 'Close' dari X_test untuk representasi singkat
        x_test_display = [
            f"{val[close_price_index]:.2f}" for val in X_test_original_scale[i][-3:]
        ]
        x_test_str = f"[..., {', '.join(x_test_display)}]"

        table_data.append(
            {
                "No.": i + 1,
                "Input Sequence (Last 3 Close)": x_test_str,
                "Harga Asli (y_test)": f"{y_test_acl_ori_scl[i]:.2f}",
                "Prediksi (LSTM)": f"{predictions[i]:.2f}",
            }
        )

    # Buat DataFrame dari data tabel
    df_table = pd.DataFrame(table_data)

    # Cetak tabel
    print(df_table.to_string(index=False))
    print("--- Tabel Selesai ---")


def create_seq_for_eval(data: np.ndarray, window_size: int, close_price_index: int):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : (i + window_size)])
        y.append(data[i + window_size, close_price_index])
    return np.array(X), np.array(y)


def main():
    core = Core(
        window_size=20,
        epochs=200,
        hidden_size=128,
        num_layers=2,
        lr=1e-4,
        batch_size=32,
        dropout=0.2,
    )

    data = core.fetch_stock_data("AAPL", "5y")

    train_data, test_data, scalers, features = core.preprocess_stock_data(data)
    close_price_index = features.index("Close")
    print(f"Train data: {len(train_data)} baris, Test data: {len(test_data)} baris")

    predictions = core.forecast_with_lstm(
        train_data, test_data, scalers, close_price_index
    )

    # Ambil y_test (skala 0-1) dari test_data
    X_test_raw_scaled, y_test_scaled = create_seq_for_eval(
        test_data, core.window_size, close_price_index
    )

    close_scaler = scalers["Close"]

    y_test_acl_ori_scl = (
        close_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten().tolist()
    )

    evaluasi_RMSE_MAE(predictions, y_test_acl_ori_scl)
    tableGenerate(
        scalers,
        features,
        X_test_raw_scaled,
        predictions,
        y_test_acl_ori_scl,
        close_price_index,
    )
    plot_predictions(y_test_acl_ori_scl, predictions)


if __name__ == "__main__":
    main()
