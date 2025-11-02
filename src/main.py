from app.core import Core
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def evaluasi_RMSE_MAE(predictions: list, y_test_actual_original_scale: list):
    print("\n--- Evaluasi Akurasi Robot LSTM ---")
    # Pastikan panjang prediksi dan harga asli sama
    # Jika tidak sama, mungkin ada masalah di create_sequences atau prediksi
    if len(predictions) != len(y_test_actual_original_scale):
        print(
            f"Peringatan: Panjang prediksi ({len(predictions)}) tidak sama dengan panjang harga asli ({len(y_test_actual_original_scale)})."
        )
        # Ambil bagian yang lebih pendek untuk perbandingan
        min_len = min(len(predictions), len(y_test_actual_original_scale))
        predictions = predictions[:min_len]
        y_test_actual_original_scale = y_test_actual_original_scale[:min_len]

    # 2. Hitung Metrik Akurasi
    rmse = np.sqrt(mean_squared_error(y_test_actual_original_scale, predictions))
    mae = mean_absolute_error(y_test_actual_original_scale, predictions)

    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
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
    scaler: MinMaxScaler,
    X_test_raw_scaled: np.ndarray,
    predictions: list,
    y_test_acl_ori_scale: list,
):
    # --- Tampilkan Tabel Perbandingan ---
    print("\n--- Tabel Perbandingan Prediksi ---")

    # X_test_original_scale akan berbentuk (jumlah_sampel, window_size, 1)
    X_test_original_scale = scaler.inverse_transform(
        X_test_raw_scaled.reshape(-1, 1)
    ).reshape(X_test_raw_scaled.shape[0], X_test_raw_scaled.shape[1], 1)

    # Buat list untuk data tabel
    table_data = []
    for i in range(len(predictions)):
        # Ambil 3 nilai terakhir dari X_test untuk representasi singkat
        # Atau bisa juga hanya nilai terakhir: f"{X_test_original_scale[i][-1][0]:.2f}"
        x_test_display = [f"{val[0]:.2f}" for val in X_test_original_scale[i][-3:]]
        x_test_str = f"[..., {', '.join(x_test_display)}]"

        table_data.append(
            {
                "No.": i + 1,
                "Input Sequence (Last 3)": x_test_str,
                "Harga Asli (y_test)": f"{y_test_acl_ori_scale[i]:.2f}",
                "Prediksi (LSTM)": f"{predictions[i]:.2f}",
            }
        )

    # Buat DataFrame dari data tabel
    df_table = pd.DataFrame(table_data)

    # Cetak tabel
    print(df_table.to_string(index=False))
    print("--- Tabel Selesai ---")


def create_seq_for_eval(data: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : (i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def main():
    core = Core()

    data = core.fetch_stock_data("AAPL", "1y")
    train_data, test_data, scaler = core.preprocess_stock_data(data)
    print(f"Train data: {len(train_data)} baris, Test data: {len(test_data)} baris")

    # Ganti pemanggilan fungsi dari GPT ke LSTM
    # predictions = core.forecast_with_gpt(train_data, test_data, scaler)
    predictions = core.forecast_with_lstm(train_data, test_data, scaler)

    # Ambil y_test (skala 0-1) dari test_data
    X_test_raw_scaled, y_test_scaled = create_seq_for_eval(test_data, core.window_size)
    # Ubah y_test_scaled kembali ke skala harga asli
    y_test_acl_ori_scl = scaler.inverse_transform(y_test_scaled).flatten().tolist()

    evaluasi_RMSE_MAE(predictions, y_test_acl_ori_scl)
    tableGenerate(scaler, X_test_raw_scaled, predictions, y_test_acl_ori_scl)
    plot_predictions(y_test_acl_ori_scl, predictions)


if __name__ == "__main__":
    main()
