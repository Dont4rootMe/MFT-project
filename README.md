# MFT-project: Reinforcement Learning для Автоматизированной Торговли Криптовалютами

## 🎯 Цель проекта

**Бизнес-цель**: Разработка и развертывание интеллектуальной торговой системы на основе обучения с подкреплением (Reinforcement Learning) для автоматизации торговли криптовалютами в общем случае на множестве монет на множестве торговых бирж. Система должна максимизировать прибыль при управлении рисками путем принятия оптимальных торговых решений в режиме реального времени парадигмы MFT торговли (с частотой от 1 раза в день, до 1 раза в несколько минут).

Данный проект содержит в себе реализованный и подогнанный под наши нужды код энвайронмента торговой площадки, для иммитации работы торговых бирж на исторических данных, своим интерфейсом совместимый с gymnassium от openai. Данный энв предлагает широкий спектр моделей поведения/интерфейсов взаимодействия обучаемой политики с рынком, покрывающий все нужды аналитиков набор логируемых метрик, настройки иммитатора трейдера, валют и торговых площадок.

**Примемы**: 
1) Обучение Convolution, Attention based агентов, обучаемых при помощи RL – A2C, PPO, DQN
2) Применение LLM в качестве регрессора состояния рынка по новостям
3) Применение байесовской оптимизации портфеля в качестве абсолютного состояния портфела, с использованием вывода агентов в качестве дельта экшенов
4) Реализация классических бейзлайнов для валидации/мультиагентной схемы работы приложения. 
5) Реализация telegram бота для мониторинга работы сервера

### Основные свойства проекта

Проект использует hydra фреймворк для организации экспериментов, разделен на логические состовляющие (pipelines, tensortrade и conf), предлагает широкий спектр графических методов валидации агентов.

### Ключевые задачи:
- Создание агентов RL (A2C, PPO, DQN) для генерации торговых сигналов
- Разработка продакшн-системы с низкой латентностью для детерминированного исполнения ордеров
- Обеспечение стабильности и надежности системы в условиях волатильности рынка
- Минимизация рисков и максимизация доходности портфеля

### Целевая аудитория:
- Криптовалютные трейдеры
- Инвестиционные фонды
- Автоматизированные торговые платформы

---

## 📊 Целевые метрики для продакшена

### 1. Производительность системы
| Метрика | Целевое значение | Критичность |
|---------|------------------|-------------|
| **Среднее время отклика модели** | ≤ 1000 мс | Критично |
| **P95 латентность** | ≤ 1200 мс | Высокая |

### 2. Надежность и доступность
| Метрика | Целевое значение | Критичность |
|---------|------------------|-------------|
| **Доля неуспешных запросов от клиента telegram** | ≤ 0.1% | Критично |
| **Uptime сервиса** | ≥ 99.99% | Критично |
| **Время автоматического восстановления (MTTR)** | ≤ 1 минут | Высокая |

### 3. Использование ресурсов
| Метрика | Целевое значение | SLA |
|---------|------------------|-----|
| **CPU utilization** | ≤ 70% | Средняя нагрузка |
| **Memory usage** | ≤ 16 GB | Пик нагрузки |
| **GPU memory (если используется)** | ≤ 24 GB | Пик нагрузки |
| **Network latency** | ≤ 50 мс | К API биржи |

### 4. Качество модели
| Метрика | Целевое значение | Описание |
|---------|------------------|----------|
| **Процент неисполняемых ордеров** | ≤ 30% | Число ордеров, не способных быть исполненными при поставленных агентом ограничениях |
| **Sharpe Ratio** | ≥ 1.5 | Отношение доходности к риску |
| **Максимальная просадка (Max Drawdown)** | ≤ 15% | Максимальное падение капитала |
| **Годовая доходность** | ≥ 20% | Среднегодовой возврат |
| **Win Rate** | ≥ 55% | Доля прибыльных сделок |
| **Profit Factor** | ≥ 1.5 | Отношение прибыли к убыткам |
| **Средняя прибыль на сделку** | ≥ 0.5% | После комиссий |

### 5. Мониторинг и наблюдаемость
| Метрика | Целевое значение |
|---------|------------------|
| **Логирование в telegram** | Мониторинг каждого действия запущенного ансамбля агентов |
| **Model performance degradation** | Alert при Sharpe < 1.0 |
| **Детекция прерывания временного ряда биржы** | Детекция изменения распределения цен на бирже для вызова Alert и снятия памяти агента |
| **Anomaly detection rate** | ≤ 2% ложных срабатываний |

---

## 📁 Набор данных

### Источники данных
**Основной источник**: Исторические данные торгов с различных бирж (например Binance, Bitfinex или Bitstamp) с доступным для фетчинга api. В том числе мы рассматриваем специализированные платформы по подписке, предлагающие специально разработанные высокоинформативные фичи. 

### Структура данных
Каждый датасет содержит следующие колонки:
- **date** - Временная метка
- **open** - Цена открытия
- **high** - Максимальная цена
- **low** - Минимальная цена
- **close** - Цена закрытия
- **volume** - Объем торгов

Датасет может иметь другие фичи, как например если мы используем api 3 лиц с уже созданными умными фичами.

### Feature Engineering
Проект использует модульную систему генерации признаков:
- **Технические индикаторы**: RSI, MACD, EMA, SMA, Bollinger Bands, ATR
- **Объемные индикаторы**: OBV, Volume Rate of Change
- **Ценовые паттерны**: Support/Resistance levels, Candlestick patterns
- **Рыночная микроструктура**: Bid-ask spread, Order book imbalance
- **Временные признаки**: Day of week, Hour of day, Market session
- **Инженерный анализ**: индикаторы фигур инженерного анализа

Конфигурация признаков: `conf/data/feature_engineering/`

### Feature Selection
Используется автоматизированный отбор признаков для уменьшения размерности:
- Фильтрация по корреляции
- Recursive Feature Elimination (RFE)
- Feature importance из tree-based моделей

Конфигурация отбора: `conf/data/feature_selection/`

### Разделение данных
- **Validation set**: 365 последних дней одним непрерывным контекстом
- **Training set**: Все остальные (конфигурируемо) дни исторических данных

### Временные характеристики
- **Таймфрейм**: default – 1 час / 1 день (конфигурируемо)
- **Период данных**: 2+ года исторических данных (зависит от доступности исторических данных для используемых монет).
- **Window size**: Скользящее окно (default – 20-60 периодов для агента)

---

## 🔬 План экспериментов

### Этап 1: Baseline модели (Недели 1-2)
**Цель**: Установить базовый уровень производительности

#### Эксперименты:
1. **Статистические стратегии**
   - Buy & Hold
   - Moving Average Crossover
   - Z-score Mean Reversion (`pipelines/z_score_mean_reversion/`)
   - Portfoliio ERC reweighter (`pipelines/portfolio_erc_static`)
   
2. **Machine Learning Baseline**
   - K-Nearest Neighbors Policy (`pipelines/knn_policy/`)
   - Random Forest Classifier
   - Logistic Regression

**Метрики для сравнения**: Sharpe Ratio, Max Drawdown, Total Return, PNL

---

### Этап 2: Разработка RL агентов (Недели 3-6)
**Цель**: Обучение и оптимизация RL моделей

#### 2.1 Policy Gradient методы
- **A2C (Advantage Actor-Critic)**
  - Конфигурация: `conf/train/a2c_train.yaml`
  - Архитектура: Shared network + Actor/Critic heads
  
- **PPO (Proximal Policy Optimization)**
  - Конфигурация: `conf/train/ppo_train.yaml`
  - Clipping parameter: 0.1-0.3


#### 2.2 Value-based методы
- **DQN (Deep Q-Network)**
  - Experience replay buffer
  - Target network updates
  - Double DQN modification

#### 2.3 Архитектуры нейронных сетей
Тестирование различных backbone моделей:
- **MLP (Multi-Layer Perceptron)** - `conf/model/backbone/mlp.yaml`
- **CNN (Convolutional Neural Network)** - `conf/model/backbone/cnn.yaml`
- **LSTM (Long Short-Term Memory)** - `notebooks/examples/use_lstm_rllib.ipynb`
- **Attention Networks** - `notebooks/examples/use_attentionnet_rllib.ipynb`

**Гиперпараметры для tuning**:
- Learning rate: [1e-5, 1e-3]
- Batch size: [32, 64, 128, 256]
- Hidden layers: [64, 128, 256, 512]
- Entropy coefficient: [0.001, 0.1]
- Discount factor (gamma): [0.95, 0.99, 0.999]

---

### Этап 3: Оптимизация и регуляризация (Недели 7-8)
**Цель**: Предотвращение переобучения и улучшение обобщающей способности

#### Техники:
1. **Dropout** (0.1-0.5)
2. **L2 regularization** (weight decay)
3. **Gradient clipping** (max norm: 0.5-5.0)
4. **Early stopping** (validation Sharpe Ratio)
5. **Data augmentation**:
   - Добавление шума к ценам
   - Временные сдвиги
   - Stochastic processes (`tensortrade/stochastic/`)

#### Эксперименты с reward shaping:
- Risk-adjusted returns
- Custom reward functions
- Multi-objective optimization (return vs risk)

---

### Этап 4: Portfolio Management (Недели 9-10)
**Цель**: Оптимизация управления портфелем

#### Стратегии:
1. **Equal Risk Contribution (ERC)**
   - Pipeline: `pipelines/portfolio_erc_static/`
   - Метод: Ковариационная матрица + ERC solver
   
2. **Multi-asset RL agent**
   - Расширение на несколько криптовалют
   - Dynamic position sizing
   
3. **Action schemes**:
   - Simple discrete actions (текущая)
   - Continuous action space
   - Multi-discrete actions (asset x size)

**Trade sizing strategies**:
- Fixed percentage: [0.5%, 1%, 5%, 10%, 20%, 30%, 40%]
- Kelly Criterion
- Volatility-based sizing

---

### Этап 5: Backtesting и валидация (Недели 11-12)
**Цель**: Тщательная проверка на исторических данных

#### Методология:
1. **Walk-forward analysis**
   - Rolling window: 1 год обучение, 3 месяца тест
   - Retraining frequency: ежемесячно
   
2. **Out-of-time validation**
   - Тест на совершенно новых данных
   - Разные рыночные режимы (bull/bear/sideways)
   
3. **Monte Carlo симуляции**
   - 1000+ симуляций с различными начальными условиями
   - Оценка worst-case сценариев
   
4. **Transaction cost analysis**
   - Учет комиссий биржи (0.1-0.5%)
   - Slippage modeling
   - Market impact

**Отчеты**: QuantStats integration для comprehensive analysis

---

### Этап 6: Продакшн оптимизация (Недели 13-14)
**Цель**: Подготовка к развертыванию

#### Оптимизации:
1. **Model optimization**
   - ONNX export для ускорения inference
   - Quantization (INT8)
   - Model pruning
   
2. **Infrastructure**
   - Containerization (Docker)
   - Orchestration (Kubernetes)
   - Model serving (TorchServe/TensorFlow Serving)
   
3. **Monitoring pipeline**
   - Prometheus + Grafana
   - Custom dashboards для торговых метрик
   - Alerting system
   
4. **CI/CD pipeline**
   - Automated testing
   - Model versioning (MLflow/W&B)
   - A/B testing framework

---

### Этап 7: Distributed training (Опционально)
**Цель**: Ускорение обучения для больших моделей

#### Подходы:
- **Data parallelism**: `accelerate` library support
- **Multiple GPUs**: Distributed training
- **Hyperparameter optimization**: Ray Tune / Optuna

---

## 🛠 Технологический стек

### Core ML/RL frameworks:
- **PyTorch** - Глубокое обучение
- **Gymnasium** - RL environment interface
- **TensorTrade** - Trading environment framework

### Data & Feature Engineering:
- **Pandas** - Обработка временных рядов
- **NumPy** - Численные вычисления
- **TA-Lib** - Технические индикаторы (опционально)

### Configuration & Experiment Management:
- **Hydra** - Управление конфигурациями
- **OmegaConf** - Конфигурационные файлы
- **DVC** - Версионирование данных и моделей, ML пайплайны

### Visualization & Analysis:
- **Plotly** - Интерактивные графики
- **Matplotlib** - Статические визуализации
- **QuantStats** - Торговая аналитика

### Production:
- **Accelerate** - Distributed training
- **ONNX** - Model optimization (planned)

---

## 🚀 Быстрый старт

### Установка зависимостей:
```bash
micromamba create -n mft_env python=3.11.13
micromamba activate mft_env

pip install -r requirements.txt
```

### Клонирование репозитория с данными и моделями:
```bash
# Клонирование репозитория
git clone <repository-url>
cd MFT-project

# Загрузка данных и моделей из DVC хранилища
dvc pull

# Воспроизведение полного пайплайна (опционально)
dvc repro
```

### Обучение A2C агента:
```bash
# Запуск на одной gpu одной ноде
python pipelines/rl_agent_policy/trainer.py

# Или через accelerate
bash train.sh

# Или через DVC пайплайн (рекомендуется)
dvc repro
```

### Конфигурация:
Все параметры настраиваются через YAML файлы в директории `conf/`:
- `conf/a2c_trainer.yaml` - Основная конфигурация
- `conf/data/` - Параметры данных и признаков
- `conf/model/` - Архитектура модели
- `conf/train/` - Параметры обучения

---

## 📦 DVC: Версионирование данных и моделей

Проект использует **DVC (Data Version Control)** для версионирования больших файлов данных и моделей, которые не подходят для хранения в Git.

### Структура хранения данных и моделей

| Тип | Расположение | Описание | DVC Tracking |
|-----|-------------|----------|--------------|
| **Сырые данные** | `data/raw/` | Исторические OHLCV данные криптовалют | `data/raw.dvc` |
| **Подготовленные данные** | `data/prepared/` | Данные после препроцессинга | Pipeline output |
| **Модели** | `outputs/model/` | Обученные модели и чекпоинты | Pipeline output |
| **Результаты оценки** | `outputs/evaluation/` | Метрики и результаты валидации | Pipeline output |
| **DVC Remote** | Настраивается в `.dvc/config` | Удалённое хранилище DVC | - |

### DVC Пайплайн

Проект определяет ML пайплайн в файле `dvc.yaml` с тремя стадиями:

```
+---------+      +-------+      +----------+
| prepare | ---> | train | ---> | evaluate |
+---------+      +-------+      +----------+
```

| Стадия | Описание | Скрипт | Outputs |
|--------|----------|--------|---------|
| **prepare** | Загрузка и подготовка данных с бирж | `pipelines/dvc/prepare.py` | `data/prepared/` |
| **train** | Обучение A2C агента | `pipelines/dvc/train.py` | `outputs/model/` |
| **evaluate** | Оценка модели и генерация метрик | `pipelines/dvc/evaluate.py` | `outputs/evaluation/metrics.json` |

### Метрики модели

После выполнения `dvc repro` или `dvc metrics show` доступны следующие метрики:

| Метрика | Описание |
|---------|----------|
| `sharpe_ratio` | Коэффициент Шарпа (риск-доходность) |
| `max_drawdown` | Максимальная просадка портфеля |
| `win_rate` | Доля прибыльных сделок |
| `total_return` | Общая доходность |
| `profit_factor` | Отношение прибыли к убыткам |
| `calmar_ratio` | Коэффициент Кальмара |
| `sortino_ratio` | Коэффициент Сортино |
| `final_net_worth` | Итоговая стоимость портфеля |

### Команды DVC

```bash
# Загрузка данных и моделей из удалённого хранилища
dvc pull

# Запуск полного пайплайна (prepare -> train -> evaluate)
dvc repro

# Запуск отдельной стадии
dvc repro train

# Отправка данных и моделей в удалённое хранилище
dvc push

# Проверка статуса пайплайна
dvc status

# Просмотр DAG пайплайна
dvc dag

# Просмотр метрик
dvc metrics show
```

### Версионирование данных

```bash
# Добавление новых данных под контроль DVC
dvc add data/raw

# Коммит изменений в Git (DVC файлы)
git add data/raw.dvc data/.gitignore
git commit -m "Update raw data"

# Отправка данных в удалённое хранилище
dvc push
```

### Переключение между версиями

```bash
# Переключение на конкретную версию (например, по git тегу или коммиту)
git checkout v1.0.0
dvc checkout

# Или одной командой
git checkout v1.0.0 && dvc pull
```

### Настройка удалённого хранилища

Просмотр текущего хранилища:
```bash
dvc remote list
```

Примеры настройки различных хранилищ:
```bash
# Локальное хранилище (для разработки)
dvc remote add -d local_storage /path/to/storage

# Amazon S3
dvc remote add -d s3_storage s3://bucket-name/path
dvc remote modify s3_storage access_key_id <access_key>
dvc remote modify s3_storage secret_access_key <secret_key>

# Google Cloud Storage
dvc remote add -d gcs_storage gs://bucket-name/path

# SSH/SFTP
dvc remote add -d ssh_storage ssh://user@host/path
```

### Файлы DVC

| Файл | Описание |
|------|----------|
| `dvc.yaml` | Определение пайплайна (стадии, зависимости, выходы) |
| `dvc.lock` | Хэши версий данных и параметров |
| `data/raw.dvc` | Трекинг сырых данных |
| `.dvc/config` | Конфигурация DVC (remote storage) |

---

## Ожидаемые результаты

### Минимальные критерии успеха:
- Sharpe Ratio > 1.5
- Max Drawdown < 15%
- Win Rate > 55%
- Inference latency < 100ms

### Оптимальные результаты:
- Sharpe Ratio > 2.0
- Max Drawdown < 10%
- Годовая доходность > 30%
- Стабильная работа в различных рыночных условиях

---

## Документация и примеры

Подробные примеры и туториалы доступны в директории `notebooks/examples/`:
- `train_and_evaluate.ipynb` - Полный цикл обучения и оценки
- `ledger_example.ipynb` - Работа с торговым журналом
- `use_lstm_rllib.ipynb` - Использование LSTM архитектуры
- `use_attentionnet_rllib.ipynb` - Attention mechanisms
- `setup_environment_tutorial.ipynb` - Настройка окружения

---

## Лицензия

Apache License 2.0

---

## Контакты

Для вопросов и предложений создавайте Issues в репозитории.

---

**Последнее обновление**: Январь 2026

