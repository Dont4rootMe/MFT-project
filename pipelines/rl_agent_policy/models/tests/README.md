# Model Output Validation Tests

Тесты для валидации выходов моделей и их преобразования для использования в API.

## Что тестируется

### 1. Архитектуры моделей ✅
- **CNNBackbone**: Сверточная нейросеть для обработки временных рядов
- **MLPBackbone**: Полносвязная сеть как альтернатива CNN
- **ActorHead**: Голова для предсказания действий (policy)
- **CriticHead**: Голова для оценки состояний (value function)

### 2. Валидация выходов ✅

#### Форматы выходов
- ✅ Правильная размерность тензоров
- ✅ Правильный тип данных (float32)
- ✅ Отсутствие NaN и inf значений
- ✅ Соответствие размера батча

#### Логиты (Actor Head)
- ✅ Конечные значения (finite)
- ✅ Неограниченные вещественные числа
- ✅ Правильная форма: `(batch_size, n_actions)`

#### Вероятности (Softmax преобразование)
- ✅ Диапазон [0, 1]
- ✅ Сумма по действиям = 1
- ✅ Численная стабильность
- ✅ Логарифмические вероятности ≤ 0

#### Действия (Action indices)
- ✅ Целочисленный тип
- ✅ Диапазон [0, n_actions-1]
- ✅ Детерминированный выбор (argmax)
- ✅ Стохастический выбор (sampling)

#### Значения (Critic Head)
- ✅ Скалярное значение на элемент батча
- ✅ Конечные вещественные числа
- ✅ Правильная форма: `(batch_size, 1)`

### 3. Преобразования для API ✅

#### Logits → Probabilities
```python
logits = actor(features)  # Raw output
probs = F.softmax(logits, dim=-1)  # [0, 1], sum to 1
```

#### Probabilities → Actions
```python
# Deterministic (greedy)
action = probs.argmax(dim=-1)

# Stochastic (sampling)
action = torch.multinomial(probs, num_samples=1).squeeze(-1)
```

#### Actions → One-Hot
```python
one_hot = F.one_hot(actions, num_classes=n_actions).float()
```

#### Value Normalization
```python
values_normalized = (values - v_min) / (v_max - v_min + 1e-8)
```

#### Confidence Score
```python
confidence = probs.max(dim=-1)[0]  # Max probability
```

### 4. End-to-End Pipeline ✅
- ✅ Observation → CNN → Actor → Action
- ✅ Observation → MLP → Actor → Action  
- ✅ Observation → CNN → Critic → Value
- ✅ Actor + Critic вместе (A2C style)

### 5. Градиенты ✅
- ✅ Градиенты проходят через все слои
- ✅ Нет NaN в градиентах
- ✅ Обучаемые параметры обновляются

### 6. Граничные случаи ✅
- ✅ Батч размера 1
- ✅ Большие батчи (128+)
- ✅ Нулевые входы
- ✅ Экстремальные значения входов

## Структура тестов

```
tests/
├── __init__.py           # Инициализация
├── conftest.py           # Фикстуры и валидаторы
├── test_models.py        # Основные тесты (39 тестов)
├── main_test.py          # Главный скрипт запуска
├── check_tests.sh        # Быстрая проверка окружения
├── run_tests.sh          # Скрипт запуска (legacy)
├── README.md             # Эта документация
└── SUMMARY.md            # Резюме тестов
```

## Запуск тестов

### Из папки tests/ (рекомендуется)

```bash
# Перейти в папку тестов
cd pipelines/rl_agent_policy/models/tests/

# Запустить все тесты
python main_test.py

# С подробным выводом
python main_test.py -v

# Конкретный файл тестов
python main_test.py -f test_models.py

# С coverage
python main_test.py --coverage

# С HTML отчетом
python main_test.py --html

# Список всех тестов
python main_test.py --list-tests

# Быстрая проверка окружения
bash check_tests.sh
```

### Из корня проекта

```bash
# Запустить все тесты
python -m pytest pipelines/rl_agent_policy/models/tests/ -v

# Конкретный тест
pytest pipelines/rl_agent_policy/models/tests/test_models.py::TestActorHead -v

# С coverage
pytest pipelines/rl_agent_policy/models/tests/ --cov=pipelines/rl_agent_policy/models --cov-report=html
```

### Быстрый запуск (legacy)
```bash
cd pipelines/rl_agent_policy/models/tests/
./run_tests.sh
```

## Примеры использования для API

### Базовый пример

```python
import torch
import torch.nn.functional as F
from pipelines.rl_agent_policy.models import CNNBackbone, ActorHead

# Инициализация
input_shape = (3, 100)  # (channels, length)
n_actions = 5

backbone = CNNBackbone(input_shape)
actor = ActorHead(backbone.output_dim, n_actions)

# Получение наблюдения
observation = torch.randn(1, *input_shape)

# Forward pass
features = backbone(observation)
logits = actor(features)

# Преобразование в вероятности
probs = F.softmax(logits, dim=-1)

# Выбор действия (детерминированный)
action = probs.argmax(dim=-1).item()

# Или стохастический
action_stochastic = torch.multinomial(probs, num_samples=1).item()

# Получение уверенности
confidence = probs.max().item()

print(f"Action: {action}")
print(f"Confidence: {confidence:.2%}")
print(f"Probabilities: {probs.squeeze()}")
```

### API Response Format

```python
def predict_action(observation):
    """API endpoint для предсказания действия."""
    # Preprocessing
    obs_tensor = preprocess_observation(observation)
    
    # Model inference
    with torch.no_grad():
        features = backbone(obs_tensor)
        logits = actor(features)
        probs = F.softmax(logits, dim=-1)
    
    # Get action and confidence
    action = probs.argmax(dim=-1).item()
    confidence = probs.max().item()
    
    # Prepare response
    response = {
        "action": int(action),
        "confidence": float(confidence),
        "probabilities": probs.squeeze().tolist(),
        "action_names": ACTION_NAMES[action]
    }
    
    return response
```

### Batch Prediction

```python
def predict_batch(observations):
    """Batch prediction для множественных наблюдений."""
    # Preprocessing
    obs_tensor = torch.stack([preprocess_observation(obs) for obs in observations])
    
    # Model inference
    with torch.no_grad():
        features = backbone(obs_tensor)
        logits = actor(features)
        probs = F.softmax(logits, dim=-1)
    
    # Get actions and confidences
    actions = probs.argmax(dim=-1).cpu().numpy()
    confidences = probs.max(dim=-1)[0].cpu().numpy()
    
    return {
        "actions": actions.tolist(),
        "confidences": confidences.tolist(),
        "batch_size": len(observations)
    }
```

## Проверяемые инварианты

### Математические свойства

1. **Вероятности**:
   - `0 ≤ P(a) ≤ 1` для всех действий a
   - `Σ P(a) = 1`

2. **Логиты**:
   - Могут быть любыми вещественными числами
   - `logit_i - logit_j = log(P(i)/P(j))`

3. **Действия**:
   - Целочисленные индексы
   - `0 ≤ action < n_actions`

4. **Значения**:
   - Вещественные числа (unbounded)
   - Обычно нормализуются для стабильности

### Численная стабильность

```python
# Плохо: может привести к overflow
probs = torch.exp(logits) / torch.exp(logits).sum()

# Хорошо: численно стабильно
probs = F.softmax(logits, dim=-1)

# Ещё лучше: работа с log-probabilities
log_probs = F.log_softmax(logits, dim=-1)
```

## Типичные ошибки и их обнаружение

### 1. NaN в выходах
```python
# Тест обнаружит
assert not torch.isnan(output).any()

# Причины:
# - Division by zero
# - Log of negative number
# - Gradient explosion
```

### 2. Неправильная сумма вероятностей
```python
# Тест обнаружит
assert torch.allclose(probs.sum(dim=-1), torch.ones(...))

# Причина: неправильное применение softmax
```

### 3. Действия вне диапазона
```python
# Тест обнаружит
assert (actions >= 0).all() and (actions < n_actions).all()

# Причина: неправильный argmax или sampling
```

### 4. Неправильная форма тензора
```python
# Тест обнаружит
assert output.shape == expected_shape

# Причина: ошибка в архитектуре сети
```

## Статистика тестов

| Категория | Тестов | Описание |
|-----------|--------|----------|
| CNNBackbone | 7 | Архитектура и выходы |
| MLPBackbone | 4 | Альтернативная архитектура |
| ActorHead | 7 | Предсказание действий |
| CriticHead | 4 | Оценка состояний |
| End-to-End | 4 | Полные пайплайны |
| Transformations | 6 | API преобразования |
| Edge Cases | 4 | Граничные случаи |
| **Всего** | **~36** | **Полное покрытие** |

## Зависимости

```bash
pip install torch>=1.10.0
pip install pytest>=7.0.0
```

## CI/CD Integration

```yaml
name: Model Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - name: Install dependencies
        run: |
          pip install torch pytest
      - name: Run tests
        run: |
          pytest pipelines/rl_agent_policy/models/tests/ -v
```

## Troubleshooting

### CUDA not available
Тесты работают и без CUDA, используется CPU fallback.

### Import errors
Убедитесь, что запускаете из корня проекта:
```bash
cd /path/to/MFT-project
python -m pytest pipelines/rl_agent_policy/models/tests/
```

### Медленные тесты
Используйте CPU для быстрых тестов:
```bash
CUDA_VISIBLE_DEVICES="" pytest pipelines/rl_agent_policy/models/tests/
```

## Расширение тестов

Для добавления новой модели:

1. Создайте класс теста в `test_models.py`
2. Добавьте валидацию выходов
3. Проверьте преобразования для API
4. Обновите этот README

## Контакты

При возникновении проблем создайте issue в репозитории.

