# Credit Score Classification Project

## 🎯 Цель проекта

Разработка и обучение моделей машинного обучения для прогнозирования кредитного рейтинга клиентов на основе различных финансовых и демографических характеристик.

## 📂 Структура проекта

- `models/` — сохранённые модели и препроцессор:
  - `model_logistic.pkl`
  - `model_tree.pkl`
  - `model_random_forest.pkl`
  - `preprocessor.pkl`
  - `label_encoder.pkl`
- `results/` — результаты и предсказания:
  - `test_predictions_labeled.csv` - результаты предсказания
- `README.md` — описание проекта;
- `RESULTS.md` - результаты проекта;
- `main.ipynb` — исследовательский анализ, предобработка, обучение моделей, визуализации;
- `preprocessing_utils.py` — функция препроцессинга;
- `test.csv` - тестовый датасет;
- `testing.ipynb` — предсказания на тестовых данных;
- `train.csv` - тренировочный датасет.

## 🗃️ Использованные данные

- **Файл**: `train.csv` (обучение), `test.csv` (предсказание)
- **Таблицы в PostgreSQL**:
  - `credit_data` — сырые данные
  - `credit_data_cleaned` — после `preprocess_credit_data`
  - `credit_features_processed` — после трансформаций и pipeline
  - `test_data` — сырые данные из `test.csv`

## 📊 Описание колонок

- `age`: возраст
- `occupation`: тип занятости
- `num_credit_card`: количество кредитных карт
- `annual_income`, `monthly_inhand_salary`, `outstanding_debt`: числовые финансовые признаки
- `credit_mix`, `payment_behaviour`, `credit_score`: категориальные признаки
- Дополнительно созданы признаки:
  - `credit_utilization_rate`
  - `credit_history_months`

## ⚙️ Алгоритмы и модели

- **Модели**:
  - `LogisticRegression`
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
- **Метрики**:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1-score (macro)
- **Pipeline**:
  - Числовые признаки: `SimpleImputer + MinMaxScaler`
  - Категориальные признаки: `SimpleImputer + OrdinalEncoder + MinMaxScaler`
  - Объединение через `ColumnTransformer`

## 📈 Визуализации

- Доля 'poor' по:
  - Возрастным группам
  - Количеству кредитных карт
  - Типу занятости
- Тепловая карта: профессия × количество кредитных карт

## 💾 Сохранение моделей и объектов

```python
with open("models/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
with open("models/model_logistic.pkl", "wb") as f:
    pickle.dump(pipe_log, f)
with open("models/model_tree.pkl", "wb") as f:
    pickle.dump(pipe_tree, f)
with open("models/model_random_forest.pkl", "wb") as f:
    pickle.dump(pipe_rf, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
```

## Работа с базой данных PostgreSQL

Для удобства хранения, обработки и анализа данных проект использует базу данных PostgreSQL. Было создано несколько таблиц:

| Название таблицы          | Описание |
|---------------------------|----------|
| `credit_data`             | Сырые необработанные данные из `train.csv` |
| `credit_data_cleaned`     | Данные после предварительной обработки (чистка и нормализация) |
| `credit_features_processed` | Данные после feature engineering и масштабирования |
| `test_data`               | Сырые данные из `test.csv` |

### 🔌 Подключение к PostgreSQL

Для подключения к базе использовалась библиотека:

```python
from sqlalchemy import create_engine
import psycopg2
```
Пример подключения:
```python
engine = create_engine("postgresql+psycopg2://postgres:qwerty@localhost:5432/postgres")
conn = engine.connect()
```
Пример выгрузки DataFrame в PostgreSQL:
```python
df_clean.to_sql("credit_data_cleaned", engine, if_exists="replace", index=False)
```
Пример загрузки данных из таблицы:
```python
data = pd.read_sql("SELECT * FROM credit_data", conn)
```
