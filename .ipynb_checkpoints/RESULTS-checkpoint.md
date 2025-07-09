# 📊 Результаты модели кредитного скоринга

## 🔁 Train/Test Split

Модели обучены на выборке `train.csv`, разделённой через `train_test_split`.

---

## 📈 Метрики моделей (train)

| Model               | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|---------------------|----------|-------------------|----------------|------------------|
| Logistic Regression | 0.6155   | 0.5972            | 0.5451         | 0.5619           |
| Decision Tree       | 0.6915   | 0.6655            | 0.6810         | 0.6717           |
| Random Forest       | 0.7162   | 0.6914            | 0.7023         | 0.6961           |

---

## 📊 Матрицы ошибок

### Random Forest

|                   | Predicted good | Predicted poor | Predicted standard |
|-------------------|----------------|----------------|---------------------|
| Actual good       | 1292           | 40             | 618                 |
| Actual poor       | 202            | 2103           | 689                 |
| Actual standard   | 696            | 776            | 4229                |

### Logistic Regression

|                   | Predicted good | Predicted poor | Predicted standard |
|-------------------|----------------|----------------|---------------------|
| Actual good       | 770            | 30             | 1150                |
| Actual poor       | 154            | 1426           | 1414                |
| Actual standard   | 485            | 860            | 4356                |

### Decision Tree

|                   | Predicted good | Predicted poor | Predicted standard |
|-------------------|----------------|----------------|---------------------|
| Actual good       | 1302           | 103            | 545                 |
| Actual poor       | 211            | 1970           | 813                 |
| Actual standard   | 786            | 826            | 4089                |

---

## 📌 SQL-анализ (PostgreSQL)

Аналитика по:
- Возрастным группам (`age_group`)
- Количеству кредитных карт (`num_credit_card`)
- Типу занятости (`occupation`)
- Соотношение профессии и числа кредитных карт — тепловая карта

---

## 🔍 Результаты теста (`test.csv`)

- Использована функция `preprocess_credit_data()` из `preprocessing_utils.py`
- Модели загружены из `.pkl` файлов
- Предсказания сохранены в `results/test_predictions_labeled.csv`

### Совпадение предсказаний моделей

- Logistic vs RandomForest: **73.90%**
- DecisionTree vs RandomForest: **85.92%**
- Logistic vs DecisionTree: **68.85%**

### Формат предсказаний

| id | pred_logistic | pred_tree | pred_rf |
|----|----------------|-----------|---------|
| 0  | good           | good      | good    |
| 1  | standard       | good      | good    |
| ...| ...            | ...       | ...     |

---

## 📤 Выгрузка результатов

Файл: `results/test_predictions_labeled.csv`

```csv
id,pred_logistic,pred_tree,pred_rf
0,good,good,good
1,standard,good,good
...
```

---

## ✅ Вывод

Модель RandomForest показала наилучшие результаты на обучающей выборке. SQL-анализ выявил важные зависимости между профессиями, возрастом и количеством кредитных карт. Финальные предсказания успешно сохранены и готовы к использованию.