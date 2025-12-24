"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def create_submission(predictions):
    """
    Создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    os.makedirs("results", exist_ok=True)
    submission_path = "results/submission.csv"
    predictions.to_csv(submission_path, index=False)

    print(f"Submission файл сохранен: {submission_path}")
    return submission_path


def main():
    """
    Главная функция программы
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    # параметры модели
    RANDOM_STATE = 322
    N_SPLITS = 5
    CONFORMAL_Q = 0.92

    NN_WEIGHT = 0.15
    SHRINK = 0.995

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # LOAD DATA
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    test = test.reset_index(drop=True)
    test["row_id"] = test.index

    # FEATURE ENGINEERING
    def create_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])
        df["day_of_year"] = df["dt"].dt.dayofyear
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)

        df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

        df["holiday_activity"] = df["holiday_flag"] * df["activity_flag"]
        df["temp_humidity"] = df["avg_temperature"] * (df["avg_humidity"] / 100.0)
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        return df

    train_fe = create_features(train)
    test_fe = create_features(test)

    # PRODUCT STATS
    product_stats = train_fe.groupby("product_id").agg({
        "n_stores": ["mean", "std"],
        "holiday_flag": "mean",
        "activity_flag": "mean"
    }).reset_index()

    product_stats.columns = ["product_id"] + [
        f"product_{a}_{b}" for a, b in product_stats.columns[1:]
    ]

    train_fe = train_fe.merge(product_stats, on="product_id", how="left")
    test_fe = test_fe.merge(product_stats, on="product_id", how="left")

    for c in product_stats.columns:
        if c != "product_id":
            test_fe[c] = test_fe[c].fillna(train_fe[c].median())

    # TARGETS
    train_fe["mid"] = 0.5 * (train_fe["price_p05"] + train_fe["price_p95"])
    train_fe["true_width"] = train_fe["price_p95"] - train_fe["price_p05"]
    train_fe["log_width"] = np.log1p(train_fe["true_width"])

    # FEATURES
    exclude = [
        "price_p05", "price_p95", "mid", "true_width", "log_width",
        "dt", "row_id", "product_id"
    ]

    FEATURES = [
        c for c in train_fe.columns
        if c not in exclude and train_fe[c].dtype != "object"
    ]

    X = train_fe[FEATURES].fillna(train_fe[FEATURES].median())
    X_test = test_fe[FEATURES].fillna(train_fe[FEATURES].median())

    groups = train_fe["product_id"]

    # LIGHTGBM MODELS
    mid_params = {
        "objective": "regression",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    }

    width_params = {
        "objective": "regression",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 5,
        "num_leaves": 32,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE + 100,
        "verbosity": -1,
    }

    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_mid = np.zeros(len(train_fe))
    oof_w = np.zeros(len(train_fe))

    mid_models, width_models = [], []

    for tr, val in gkf.split(X, train_fe["mid"], groups):
        m_mid = lgb.LGBMRegressor(**mid_params)
        m_mid.fit(
            X.iloc[tr], train_fe["mid"].iloc[tr],
            eval_set=[(X.iloc[val], train_fe["mid"].iloc[val])],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof_mid[val] = m_mid.predict(X.iloc[val])
        mid_models.append(m_mid)

        m_w = lgb.LGBMRegressor(**width_params)
        m_w.fit(
            X.iloc[tr], train_fe["log_width"].iloc[tr],
            eval_set=[(X.iloc[val], train_fe["log_width"].iloc[val])],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof_w[val] = m_w.predict(X.iloc[val])
        width_models.append(m_w)

    # PRODUCT_IDX
    pid_map = {pid: i for i, pid in enumerate(train_fe["product_id"].unique())}
    train_fe["product_idx"] = train_fe["product_id"].map(pid_map)
    test_fe["product_idx"] = test_fe["product_id"].map(pid_map).fillna(0).astype(int)

    # RESIDUAL NN
    class ResidualNet(nn.Module):
        def __init__(self, d_in, n_prod, emb=16):
            super().__init__()
            self.emb = nn.Embedding(n_prod, emb)
            self.net = nn.Sequential(
                nn.Linear(d_in + emb, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.mid = nn.Linear(64, 1)
            self.w = nn.Linear(64, 1)

        def forward(self, x, pid):
            e = self.emb(pid)
            h = self.net(torch.cat([x, e], 1))
            return self.mid(h), self.w(h)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs_test = scaler.transform(X_test)

    dataset = TensorDataset(
        torch.FloatTensor(Xs),
        torch.LongTensor(train_fe["product_idx"].values),
        torch.FloatTensor(train_fe["mid"].values - oof_mid).unsqueeze(1),
        torch.FloatTensor(train_fe["log_width"].values - oof_w).unsqueeze(1)
    )

    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model_nn = ResidualNet(X.shape[1], len(pid_map))
    opt = optim.AdamW(model_nn.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss()

    for _ in range(30):
        for xb, pb, ym, yw in loader:
            opt.zero_grad()
            pm, pw = model_nn(xb, pb)
            (loss_fn(pm, ym) + 0.7 * loss_fn(pw, yw)).backward()
            opt.step()

    # TEST PREDICTION
    test_mid = np.mean([m.predict(X_test) for m in mid_models], axis=0)
    test_w = np.expm1(np.mean([m.predict(X_test) for m in width_models], axis=0))

    with torch.no_grad():
        dm, dw = model_nn(
            torch.FloatTensor(Xs_test),
            torch.LongTensor(test_fe["product_idx"].values)
        )

    test_mid += NN_WEIGHT * dm.numpy().flatten()
    test_w += NN_WEIGHT * np.expm1(dw.numpy().flatten())

    # CONFORMAL
    q = np.quantile(np.abs(train_fe["mid"] - oof_mid), CONFORMAL_Q)
    test_w = 0.7 * test_w + 0.3 * (2 * q)

    # FINAL INTERVAL
    price_p05 = test_mid - test_w / 2
    price_p95 = test_mid + test_w / 2

    mid = 0.5 * (price_p05 + price_p95)
    half = 0.5 * (price_p95 - price_p05) * SHRINK

    price_p05 = mid - half
    price_p95 = mid + half
    price_p95 = np.maximum(price_p95, price_p05 + 1e-3)

    predictions = pd.DataFrame({
        "row_id": test_fe["row_id"],
        "price_p05": price_p05,
        "price_p95": price_p95
    })

    # SAVE SUBMISSION
    create_submission(predictions)

    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()

