from app.core import Core
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


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
    scaler: MinMaxScaler,
    X_test_raw_scaled: np.ndarray,
    predictions: list,
    y_test_acl_ori_scl: list,
    close_price_index: int,
):
    # --- Tampilkan Tabel Perbandingan ---
    print("\n--- Tabel Perbandingan Prediksi ---")

    num_samples, window_size, num_features = X_test_raw_scaled.shape
    X_test_reshaped_for_scaler = X_test_raw_scaled.reshape(-1, num_features)
    X_test_original_scale_flat = scaler.inverse_transform(X_test_reshaped_for_scaler)
    X_test_original_scale = X_test_original_scale_flat.reshape(
        num_samples, window_size, num_features
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
        window_size=10,
        epochs=200,
        hidden_size=64,
        num_layers=1,
        lr=1e-3,
        batch_size=32,
        dropout=0.1,
    )

    data = core.fetch_stock_data("AAPL", "5y")
    close_price_index = data.columns.get_loc("Close")
    if close_price_index == -1:
        raise ValueError("Kolom 'Close' tidak ditemukan dalam data saham.")

    train_data, test_data, scaler = core.preprocess_stock_data(data)
    print(f"Train data: {len(train_data)} baris, Test data: {len(test_data)} baris")

    # Ganti pemanggilan fungsi dari GPT ke LSTM
    # predictions = core.forecast_with_gpt(train_data, test_data, scaler)
    predictions = core.forecast_with_lstm(
        train_data, test_data, scaler, close_price_index
    )

    # Ambil y_test (skala 0-1) dari test_data
    X_test_raw_scaled, y_test_scaled = create_seq_for_eval(
        test_data, core.window_size, close_price_index
    )

    # 1. Buat array dummy dengan shape (jumlah_sampel, jumlah_fitur_asli_scaler)
    dummy_array_for_y_test = np.zeros((len(y_test_scaled), scaler.n_features_in_))

    # 2. Masukkan y_test_scaled (yang sudah 1D) ke kolom 'Close' yang benar
    dummy_array_for_y_test[:, close_price_index] = y_test_scaled.flatten()

    # 3. Lakukan inverse transform pada array dummy, lalu ambil kembali kolom 'Close'
    y_test_acl_ori_scl = scaler.inverse_transform(dummy_array_for_y_test)[
        :, close_price_index
    ].tolist()

    evaluasi_RMSE_MAE(predictions, y_test_acl_ori_scl)
    tableGenerate(
        scaler, X_test_raw_scaled, predictions, y_test_acl_ori_scl, close_price_index
    )
    plot_predictions(y_test_acl_ori_scl, predictions)


if __name__ == "__main__":
    main()
