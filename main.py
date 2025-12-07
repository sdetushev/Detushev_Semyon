"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import os
import re
import gc
import random
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sentence_transformers import SentenceTransformer


warnings.filterwarnings("ignore")

# Параметры
SEED = 993
EMB_BATCH = 256
PCA_COMPONENTS = 8
TFIDF_TITLE_MAX_FEAT = 1200
TFIDF_DESC_MAX_FEAT = 800
USE_SBERT = True

# Фиксируем сид
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    import os
    import pandas as pd
    
    # Создать пандас таблицу submission
    if isinstance(predictions, pd.DataFrame):
        submission = predictions
    else:
        # Если predictions - это только значения, нужно создать DataFrame
        # В нашем случае predictions будет DataFrame с колонками ['id', 'prediction']
        submission = predictions
    
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


# Очистка текста
def simple_text_clean(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


# Загрузка данных
def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    for col in ['query', 'product_title', 'product_description', 'product_bullet_point', 
                'product_brand', 'product_color', 'product_locale']:
        if col in train.columns:
            train[col] = train[col].fillna('')
        if col in test.columns:
            test[col] = test[col].fillna('')
    
    return train, test


# Создание эмбеддингов с использованием SentenceTransformer
def make_sentence_embeddings(texts: List[str], model, batch_size=EMB_BATCH):
    if model is None:
        return np.zeros((len(texts), 1), dtype=np.float32)
    
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return emb


# Вычисление сходств между парами эмбеддингов
def compute_pair_similarities(query_emb, title_emb):
    q = np.asarray(query_emb, dtype=np.float32)
    t = np.asarray(title_emb, dtype=np.float32)
    
    q_norm = np.linalg.norm(q, axis=1) + 1e-12
    t_norm = np.linalg.norm(t, axis=1) + 1e-12
    
    dot = np.sum(q * t, axis=1)
    cosine = dot / (q_norm * t_norm)
    
    euclid = np.linalg.norm(q - t, axis=1)
    euclid_sim = 1.0 / (1.0 + euclid)
    
    manh = np.sum(np.abs(q - t), axis=1)
    manh_sim = 1.0 / (1.0 + manh)
    
    prod = q * t
    absdiff = np.abs(q - t)
    prod_mean = prod.mean(axis=1)
    prod_std = prod.std(axis=1)
    absdiff_mean = absdiff.mean(axis=1)
    absdiff_std = absdiff.std(axis=1)
    
    return {
        'neural_cosine': cosine,
        'neural_euclid_sim': euclid_sim,
        'neural_manhattan_sim': manh_sim,
        'neural_prod_mean': prod_mean,
        'neural_prod_std': prod_std,
        'neural_absdiff_mean': absdiff_mean,
        'neural_absdiff_std': absdiff_std,
        'neural_q_norm': q_norm,
        'neural_t_norm': t_norm,
        'neural_dot': dot
    }


# Создание базовых признаков
def create_simple_features(df):
    df['query_clean'] = df['query'].apply(simple_text_clean)
    df['title_clean'] = df['product_title'].apply(simple_text_clean)
    df['desc_clean'] = df['product_description'].apply(simple_text_clean)
    
    df['query_len'] = df['query_clean'].str.len().fillna(0).astype(int)
    df['title_len'] = df['title_clean'].str.len().fillna(0).astype(int)
    df['desc_len'] = df['desc_clean'].str.len().fillna(0).astype(int)
    df['query_words'] = df['query_clean'].str.split().str.len().fillna(0).astype(int)
    df['title_words'] = df['title_clean'].str.split().str.len().fillna(0).astype(int)
    
    if 'product_brand' in df.columns:
        df['has_brand'] = (~df['product_brand'].isna()).astype(int)
    else:
        df['has_brand'] = 0
    
    if 'product_color' in df.columns:
        df['has_color'] = (~df['product_color'].isna()).astype(int)
    else:
        df['has_color'] = 0
    
    df['has_description'] = (df['desc_len'] > 10).astype(int)
    
    def word_match_share(row):
        qwords = set(str(row['query_clean']).split())
        twords = set(str(row['title_clean']).split())
        if not qwords:
            return 0
        return len(qwords.intersection(twords)) / (len(qwords) + 1e-12)
    
    df['word_match'] = df.apply(word_match_share, axis=1)
    
    def jaccard_sim(row):
        qwords = set(str(row['query_clean']).split())
        twords = set(str(row['title_clean']).split())
        if not qwords and not twords:
            return 0.0
        intersection = len(qwords.intersection(twords))
        union = len(qwords.union(twords))
        return intersection / (union + 1e-12)
    
    df['jaccard_sim'] = df.apply(jaccard_sim, axis=1)
    
    df['query_has_dollar'] = df['query'].str.contains(r'\$', na=False).astype(int)
    df['query_has_hash'] = df['query'].str.contains(r'#', na=False).astype(int)
    df['query_has_numbers'] = df['query'].str.contains(r'\d', na=False).astype(int)
    
    return df


# Создание TF-IDF признаков
def create_tfidf_features(train, test):
    vectorizer_title = TfidfVectorizer(
        max_features=TFIDF_TITLE_MAX_FEAT, 
        stop_words='english', 
        ngram_range=(1, 3), 
        min_df=2
    )
    vectorizer_title.fit(train['query_clean'] + " " + train['title_clean'])
    
    def cosine_pair(df, vec):
        qv = vec.transform(df['query_clean'])
        tv = vec.transform(df['title_clean'])
        q_sq = qv.multiply(qv).sum(axis=1).A1
        t_sq = tv.multiply(tv).sum(axis=1).A1
        qt_dot = qv.multiply(tv).sum(axis=1).A1
        denom = np.sqrt(q_sq * t_sq) + 1e-12
        cos = qt_dot / denom
        cos[np.isnan(cos)] = 0.0
        return cos
    
    train['cosine_sim'] = cosine_pair(train, vectorizer_title)
    test['cosine_sim'] = cosine_pair(test, vectorizer_title)
    
    vectorizer_desc = TfidfVectorizer(
        max_features=TFIDF_DESC_MAX_FEAT, 
        stop_words='english', 
        ngram_range=(1, 2), 
        min_df=3
    )
    vectorizer_desc.fit(train['desc_clean'])
    
    train_desc_vec = vectorizer_desc.transform(train['desc_clean'])
    test_desc_vec = vectorizer_desc.transform(test['desc_clean'])
    
    vtq_for_desc = TfidfVectorizer(vocabulary=vectorizer_desc.vocabulary_)
    train_qt_desc_space = vtq_for_desc.fit_transform((train['query_clean'] + " " + train['title_clean']).tolist())
    test_qt_desc_space = vtq_for_desc.transform((test['query_clean'] + " " + test['title_clean']).tolist())
    
    def sparse_cosine(a, b):
        a_sq = a.multiply(a).sum(axis=1).A1
        b_sq = b.multiply(b).sum(axis=1).A1
        dot = a.multiply(b).sum(axis=1).A1
        denom = np.sqrt(a_sq * b_sq) + 1e-12
        cos = dot / denom
        cos[np.isnan(cos)] = 0.0
        return cos
    
    train['desc_cosine'] = sparse_cosine(train_desc_vec, train_qt_desc_space)
    test['desc_cosine'] = sparse_cosine(test_desc_vec, test_qt_desc_space)
    
    return train, test


# Создание нейросетевых признаков
def create_neural_features(train, test, model=None):
    if model is None:
        emb_dim = 1
        train_q_emb = np.zeros((len(train), emb_dim))
        train_t_emb = np.zeros((len(train), emb_dim))
        test_q_emb = np.zeros((len(test), emb_dim))
        test_t_emb = np.zeros((len(test), emb_dim))
    else:
        train_q_emb = make_sentence_embeddings(train['query_clean'].tolist(), model)
        train_t_emb = make_sentence_embeddings(train['title_clean'].tolist(), model)
        test_q_emb = make_sentence_embeddings(test['query_clean'].tolist(), model)
        test_t_emb = make_sentence_embeddings(test['title_clean'].tolist(), model)
    
    train_sim = compute_pair_similarities(train_q_emb, train_t_emb)
    test_sim = compute_pair_similarities(test_q_emb, test_t_emb)
    
    for k, arr in train_sim.items():
        train[f'neural_{k}'] = arr
    for k, arr in test_sim.items():
        test[f'neural_{k}'] = arr
    
    train_diff = train_q_emb - train_t_emb
    test_diff = test_q_emb - test_t_emb
    
    try:
        n_comp = min(PCA_COMPONENTS, train_diff.shape[1])
        if n_comp >= 1:
            pca = PCA(n_components=n_comp, random_state=SEED)
            pca.fit(train_diff)
            train_diff_pca = pca.transform(train_diff)
            test_diff_pca = pca.transform(test_diff)
            for i in range(train_diff_pca.shape[1]):
                train[f'neural_diff_pca_{i}'] = train_diff_pca[:, i]
                test[f'neural_diff_pca_{i}'] = test_diff_pca[:, i]
    except Exception:
        for i in range(PCA_COMPONENTS):
            train[f'neural_diff_pca_{i}'] = 0.0
            test[f'neural_diff_pca_{i}'] = 0.0
    
    return train, test


# Создание групповых признаков
def create_group_features(df):
    numeric_features = ['title_len', 'cosine_sim', 'word_match', 'jaccard_sim']
    neural_features = [c for c in df.columns if c.startswith('neural_') and df[c].dtype != 'object']
    all_features = numeric_features + neural_features
    
    agg_dict = {feat: ['mean', 'std', 'max'] for feat in all_features}
    agg_dict.update({'has_brand': ['mean'], 'has_color': ['mean']})
    
    group_stats = df.groupby('query_id').agg(agg_dict).fillna(0)
    group_stats.columns = [f'group_{col[0]}_{col[1]}' for col in group_stats.columns]
    group_stats = group_stats.reset_index()
    
    df = df.merge(group_stats, on='query_id', how='left')
    
    for feat in numeric_features + neural_features:
        mean_col = f'group_{feat}_mean'
        if mean_col in df.columns:
            df[f'{feat}_vs_group'] = df[feat] / (df[mean_col] + 1e-12)
    
    return df


# Кодирование категориальных признаков
def encode_categorical(train, test):
    categorical_cols = ['product_brand', 'product_color', 'product_locale']
    for col in categorical_cols:
        all_vals = pd.concat([train[col].fillna('unknown'), test[col].fillna('unknown')], axis=0)
        enc = LabelEncoder()
        enc.fit(all_vals)
        train[f'{col}_encoded'] = enc.transform(train[col].fillna('unknown'))
        test[f'{col}_encoded'] = enc.transform(test[col].fillna('unknown'))
    
    return train, test


# Получение списка признаков для модели
def get_feature_columns(train_df):
    neural = [c for c in train_df.columns if c.startswith('neural_')]
    pca_feats = [c for c in train_df.columns if c.startswith('neural_diff_pca_')]
    
    base = [
        'query_len', 'title_len', 'desc_len', 'query_words', 'title_words',
        'cosine_sim', 'word_match', 'jaccard_sim',
        'has_brand', 'has_color', 'has_description',
        'query_has_dollar', 'query_has_hash', 'query_has_numbers',
        'product_brand_encoded', 'product_color_encoded', 'product_locale_encoded',
        'desc_cosine'
    ]
    
    group = [c for c in train_df.columns if c.startswith('group_')]
    feats = neural + pca_feats + base + group
    feats = [f for f in feats if f in train_df.columns]
    
    seen = set()
    out = []
    for f in feats:
        if f not in seen:
            out.append(f)
            seen.add(f)
    
    return out


# Расчет nDCG@10 с группировкой
def ndcg_at_k_groupped(y_true, y_pred, group_ids, k=10):
    df = pd.DataFrame({'group': group_ids, 'y_true': y_true, 'y_pred': y_pred})
    scores = []
    
    for g, sub in df.groupby('group'):
        sub_sorted = sub.sort_values('y_pred', ascending=False).head(k)
        gains = (2 ** sub_sorted['y_true'] - 1).values
        discounts = 1.0 / np.log2(np.arange(1, len(gains) + 1) + 1)
        dcg = np.sum(gains * discounts)
        
        ideal = np.sort((2 ** sub_sorted['y_true'] - 1))[::-1]
        idcg = np.sum(ideal * discounts)
        
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    
    return np.mean(scores) if len(scores) > 0 else 0.0


# Обучение LightGBM с кросс-валидацией
def train_lgb_cv(train_df, features, target_col='relevance', n_splits=5):
    X = train_df[features].values
    y = train_df[target_col].values
    groups = train_df['query_id'].values
    
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(train_df), dtype=float)
    models = []
    
    lgb_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 10,
        'min_data_in_leaf': 64,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'min_gain_to_split': 0.1,
        'verbosity': -1,
        'seed': SEED,
        'feature_fraction_seed': SEED,
        'bagging_seed': SEED,
        'force_row_wise': True
    }
    
    fold = 0
    for train_idx, valid_idx in gkf.split(X, y, groups):
        fold += 1
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]
        groups_tr = train_df.iloc[train_idx].groupby('query_id').size().values
        groups_val = train_df.iloc[valid_idx].groupby('query_id').size().values
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, group=groups_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, group=groups_val)
        
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dtrain, dvalid],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(period=100)]
        )
        models.append(model)
        
        oof_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof[valid_idx] = oof_preds
        
        del dtrain, dvalid, X_tr, X_val, y_tr, y_val
        gc.collect()
    
    cv_score = ndcg_at_k_groupped(train_df['relevance'].values, oof, train_df['query_id'].values, k=10)
    print(f"CV nDCG@10: {cv_score:.6f}")
    
    return models, cv_score, oof


# Основной пайплайн обработки и обучения
def run_pipeline():
    print("Загрузка данных")
    train, test = load_data()
    
    print("Создание базовых признаков")
    train = create_simple_features(train)
    test = create_simple_features(test)
    
    print("Создание TF-IDF признаков")
    train, test = create_tfidf_features(train, test)
    
    model = None
    if USE_SBERT:
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("SentenceTransformer загружен успешно")
        except Exception as e:
            print(f"Не удалось загрузить SentenceTransformer: {e}")
            model = None
    
    print("Создание нейросетевых признаков")
    train, test = create_neural_features(train, test, model=model)
    
    print("Создание групповых признаков")
    train = create_group_features(train)
    test = create_group_features(test)
    
    print("Кодирование категориальных признаков")
    train, test = encode_categorical(train, test)
    
    feature_cols = get_feature_columns(train)
    available_features = [f for f in feature_cols if f in train.columns]
    print(f"Используется признаков: {len(available_features)}")
    
    print("Обучение модели LightGBM")
    models, cv_score, oof_preds = train_lgb_cv(train, available_features, target_col='relevance', n_splits=5)
    
    print("Предсказание на тестовых данных")
    X_test = test[available_features].values
    preds = np.zeros(X_test.shape[0], dtype=float)
    
    for m in models:
        preds += m.predict(X_test, num_iteration=m.best_iteration)
    preds /= len(models)
    
    # Создаем DataFrame с предсказаниями
    submission = pd.DataFrame({'id': test['id'], 'prediction': preds})
    
    print(f"Диапазон предсказаний: [{preds.min():.6f}, {preds.max():.6f}]")
    
    return submission


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Запускаем основной пайплайн
    predictions = run_pipeline()
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    submission_path = create_submission(predictions)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print(f"Файл предсказаний сохранен: {submission_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()