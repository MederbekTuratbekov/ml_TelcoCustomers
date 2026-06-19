# Customer Churn Prediction API

> Random Forest скорит каждого абонента по риску оттока за < 50 мс —
> retention-команда получает приоритизированный список до того,
> как клиент ушёл, а не после.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)]()
[![F1](https://img.shields.io/badge/F1-0.76-brightgreen)]()
[![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.85-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Проблема

Привлечение нового абонента стоит в 5–7 раз дороже удержания существующего.
Большинство операторов действуют реактивно — после факта ухода.
Эта модель скорит каждого активного абонента по риску оттока на основе условий контракта,
биллинга и использования сервисов — позволяя retention-команде вмешаться до,
а не после потери клиента.

---

## Быстрый старт

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-api
cd customer-churn-api
pip install -r requirements.txt

jupyter notebook churn_model.ipynb   # обучение
python main.py                        # API → http://127.0.0.1:8000
```

---

## Demo

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "MonthlyCharges": 95.5,
    "TotalCharges": 286.5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No"
  }'
```

```json
{
  "churn": "Yes",
  "probability": 0.87,
  "risk_tier": "high"
}
```

> `Contract`: `"Month-to-month"` / `"One year"` / `"Two year"`
> `InternetService`: `"DSL"` / `"Fiber optic"` / `"No"`
> `OnlineSecurity` / `TechSupport`: `"No"` / `"Yes"` / `"No internet service"`

---

## Результаты

| Модель                | Accuracy | F1   | ROC-AUC | Recall (churn) |
|-----------------------|----------|------|---------|----------------|
| Logistic Regression   | 79%      | 0.72 | 0.81    | 0.68           |
| Decision Tree         | 76%      | 0.69 | 0.76    | 0.71           |
| **Random Forest**     | **81%**  | **0.76** | **0.85** | **0.74**   |

**Почему F1 и ROC-AUC как primary метрики, а не Accuracy:**
Датасет несбалансирован (73% retained / 27% churned).
Модель с Accuracy=73% — предсказывающая "No churn" для всех — бесполезна для retention.
F1 и Recall на классе `Yes` измеряют то, что важно бизнесу: сколько реальных уходов модель поймала.

**Почему Random Forest, а не XGBoost:**
Прирост XGBoost на этом датасете (~1% F1) не оправдывает усложнение пайплайна и интерпретируемости.
Random Forest с `n_estimators=100` даёт feature importance без дополнительных инструментов.

---

## Датасет

- **Источник:** IBM Telco Customer Churn Dataset (Kaggle)
- **Объём:** 7 043 записи → 7 032 после удаления строк с пустым `TotalCharges`
- **Признаки:** 19 исходных → 30+ бинарных после OHE
- **Дисбаланс:** 73% retained / 27% churned — решён через `stratify=y` в сплите

| Проблема данных                         | Решение                                              | Эффект               |
|-----------------------------------------|------------------------------------------------------|----------------------|
| `TotalCharges` хранится как строка с пробелами | `replace(' ', np.nan)` → `float` → drop 11 строк | Нет silent type error в scaler |
| Дисбаланс классов 73/27                 | `stratify=y` в `train_test_split`                    | Recall: 0.61 → 0.74  |
| OHE-схема train vs inference различается | Жёстко закодированные категории в API-роутере      | Нет column mismatch  |

---

## Архитектура
    
    POST /predict   {"tenure": 3, "MonthlyCharges": 95.5, ...}

│

▼

    Pydantic v2         ← валидация типов + допустимых значений

│

▼

    OHE reconstruction  ← фиксированные категории Contract / InternetService /
    
    OnlineSecurity / TechSupport (не pd.get_dummies)

│

▼

    StandardScaler      ← загружается из scaler_TelcoCustomers.pkl

│

▼

    Random Forest       ← загружается из model_rf_TelcoCustomers.pkl

    (n_estimators=100)

│

▼

    predict_proba()     ← вероятность класса "Yes"

│

▼

    risk_tier           ← high (≥0.7) / medium (0.4–0.7) / low (<0.4)

│

▼

    {"churn": "Yes", "probability": 0.87, "risk_tier": "high"}


**Ключевые решения:**

`pd.get_dummies()` во время обучения не гарантирует порядок колонок при инференсе →
жёстко закодировал бинарные векторы для 4 категориальных признаков в API-роутере.
Нулевых ошибок column mismatch за всё время тестирования.

`StandardScaler` подогнан только на `X_train` — не на полном датасете.
Предотвращает утечку статистики теста в обучение.

---

## API

| Метод | Эндпоинт    | Вход              | Выход                                             |
|-------|-------------|-------------------|--------------------------------------------------|
| POST  | `/predict`  | 7 полей JSON      | `{churn, probability, risk_tier}`                |
| GET   | `/health`   | —                 | `{status, model, version}`                       |

`risk_tier` — готовый сигнал для CRM без дополнительной логики:
- `high` → персональный звонок от retention-агента
- `medium` → автоматизированное предложение по скидке
- `low` → стандартный NPS-опрос

---

## Стек

| Слой       | Технологии                              |
|------------|-----------------------------------------|
| ML         | scikit-learn, joblib                    |
| API        | FastAPI, Uvicorn, Pydantic              |
| Данные     | pandas, NumPy                           |
| Визуализация | Matplotlib, Seaborn                   |
| Deploy     | Local / Docker-ready                    |

---

## Что дальше (Roadmap)

- [ ] `SHAP` values в ответе — объяснение конкретного предсказания для агента
      (`"tenure=3 увеличивает риск на +0.21"`)
- [ ] Батч-эндпоинт `POST /predict/batch` — скоринг всей базы абонентов за один запрос
- [ ] MLflow: версионирование модели, трекинг экспериментов
- [ ] Data drift мониторинг (evidently) — алерт при смещении распределения признаков
- [ ] Docker Compose + Nginx для production-деплоя
- [ ] Retraining pipeline — автообновление при накоплении 1K новых размеченных примеров

---

## Business Impact

| Сценарий                                | До                              | После                           |
|-----------------------------------------|---------------------------------|---------------------------------|
| Обнаружение at-risk абонентов           | После факта ухода               | За N дней до, с вероятностью    |
| Приоритизация retention-команды         | Ручной отбор или случайный      | `risk_tier` из API              |
| Targeting retention-кампании            | Все абоненты одинаково          | High-risk → звонок, medium → оффер |
| Стоимость привлечения vs удержания      | Привлечение в 5–7× дороже       | Предотвращённый отток экономит разницу |
| Интеграция в CRM                        | Кастомная разработка            | Один REST-вызов при renewal-событии |

---

[//]: # (## Автор)
[//]: # (**[Имя]** — [LinkedIn]&#40;https://linkedin.com/in/you&#41; | [GitHub]&#40;https://github.com/you&#41;)

