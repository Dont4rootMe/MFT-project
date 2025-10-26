# Data Processing Tests

Комплексный набор тестов для модуля обработки данных криптовалют.

## Структура тестов

```
tests/
├── __init__.py                  # Инициализация пакета тестов
├── conftest.py                  # Фикстуры и вспомогательные функции
├── main_test.py                 # Главный скрипт запуска тестов
├── test_data_parser.py          # Тесты для DataHandler
├── test_feature_engine.py       # Тесты для FeatureEngineeringProcessor
├── test_feature_selection.py    # Тесты для FeatureSelector
└── test_data_validation.py      # Интеграционные тесты валидации
```

## Покрытие тестов

### test_data_parser.py
- ✅ Инициализация DataHandler
- ✅ Загрузка и подготовка данных
- ✅ Валидация OHLCV данных
- ✅ Объединение мультивалютных данных
- ✅ Кэширование
- ✅ Обработка ошибок

### test_feature_engine.py
- ✅ Генерация пользовательских признаков (MA, EMA, волатильность, RSI, MACD)
- ✅ Технические индикаторы
- ✅ Quantstats признаки
- ✅ Очистка данных
- ✅ Префиксация признаков
- ✅ Валидация значений

### test_feature_selection.py
- ✅ Фильтрация по дисперсии
- ✅ Фильтрация по корреляции
- ✅ PCA редукция
- ✅ Top-K отбор
- ✅ Per-currency и post-merge селекция
- ✅ Целостность данных

### test_data_validation.py
- ✅ Валидация формата данных
- ✅ Валидация структуры
- ✅ Валидация типов данных
- ✅ Валидация диапазонов значений
- ✅ Проверка обязательных признаков
- ✅ OHLCV консистентность
- ✅ End-to-end тесты пайплайна
- ✅ Метрики качества данных

## Установка зависимостей

```bash
# Основные зависимости для тестов
pip install pytest pytest-cov pytest-html

# Зависимости для модуля data
pip install pandas numpy scikit-learn
pip install pandas-ta quantstats  # опционально
```

## Запуск тестов

### Базовый запуск

```bash
# Запустить все тесты
python pipelines/rl_agent_policy/data/tests/main_test.py

# Или напрямую через pytest
cd /path/to/MFT-project
pytest pipelines/rl_agent_policy/data/tests/
```

### Расширенные опции

```bash
# Подробный вывод
python main_test.py -v

# Запуск конкретного файла тестов
python main_test.py -f test_data_parser.py

# С отчетом о покрытии кода
python main_test.py --coverage

# Генерация HTML отчета
python main_test.py --html

# Только быстрые тесты (без медленных интеграционных)
python main_test.py --fast

# Комбинация опций
python main_test.py -v --coverage --html
```

### Запуск конкретных тестов

```bash
# Запуск конкретного класса тестов
pytest pipelines/rl_agent_policy/data/tests/test_data_parser.py::TestDataHandlerInitialization

# Запуск конкретного теста
pytest pipelines/rl_agent_policy/data/tests/test_data_parser.py::TestDataHandlerInitialization::test_init_with_minimal_config

# Запуск тестов по маркеру
pytest -m "not slow" pipelines/rl_agent_policy/data/tests/
```

### Список всех тестов

```bash
# Показать все доступные тесты без запуска
python main_test.py --list-tests
```

## Проверяемые аспекты

### 1. Соответствие формата и структуры
- DataFrame структура
- Наличие обязательных колонок (date, OHLCV)
- Формат дат
- Отсутствие дубликатов
- Сортировка по времени

### 2. Правильность типов данных
- Числовые колонки (float/int)
- Строковые колонки (date)
- Отсутствие смешанных типов

### 3. Диапазоны значений
- Цены >= 0
- Объем >= 0
- RSI в [0, 100]
- Волатильность >= 0
- High >= Close >= Low
- High >= Open >= Low

### 4. Наличие необходимых признаков
- Базовые OHLCV признаки
- Сгенерированные признаки (при включении feature engineering)
- Префиксы для мультивалютных данных

### 5. Отсутствие невалидных значений
- Нет бесконечных значений (inf, -inf)
- Минимум NaN значений
- Корректная обработка пропусков

## Интерпретация результатов

### Успешный запуск
```
================================ test session starts =================================
collected 150 items

test_data_parser.py ..........................................          [ 28%]
test_feature_engine.py .......................................          [ 53%]
test_feature_selection.py ....................................          [ 78%]
test_data_validation.py ......................................          [100%]

================================ 150 passed in 45.23s ================================
✓ All tests passed!
```

### С покрытием кода
```
---------- coverage: platform linux, python 3.11.0 -----------
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
data_parser.py                  180     12    93%   234-245
feature_engine.py               210     18    91%   156-173
feature_selection.py            130      8    94%   98-105
-----------------------------------------------------------
TOTAL                           520     38    93%

Coverage report saved to: htmlcov/index.html
```

## Отладка упавших тестов

### Подробная информация об ошибках
```bash
# Показать локальные переменные при ошибках
pytest -vv --showlocals pipelines/rl_agent_policy/data/tests/

# Остановиться на первой ошибке
pytest -x pipelines/rl_agent_policy/data/tests/

# Запустить только упавшие тесты из предыдущего запуска
pytest --lf pipelines/rl_agent_policy/data/tests/
```

### Дебаггинг конкретного теста
```bash
# Добавить print statements и запустить с -s
pytest -s pipelines/rl_agent_policy/data/tests/test_data_parser.py::test_name

# Использовать pdb для отладки
pytest --pdb pipelines/rl_agent_policy/data/tests/
```

## CI/CD Integration

Пример для GitHub Actions:

```yaml
name: Data Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          python pipelines/rl_agent_policy/data/tests/main_test.py --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Расширение тестов

### Добавление новых тестов

1. Создайте новый тестовый класс:
```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_feature_works(self, sample_data):
        """Test that new feature works correctly."""
        result = process_feature(sample_data)
        assert result is not None
```

2. Используйте существующие фикстуры из `conftest.py`
3. Добавьте валидационные функции при необходимости

### Добавление новых фикстур

В `conftest.py`:
```python
@pytest.fixture
def custom_data():
    """Generate custom test data."""
    return create_custom_data()
```

## Рекомендации

1. **Запускайте тесты перед коммитом**: `python main_test.py --fast`
2. **Проверяйте покрытие**: Стремитесь к >90% покрытию кода
3. **Используйте моки**: Для внешних зависимостей (API, файлы)
4. **Тестируйте граничные случаи**: Пустые данные, минимальные данные, некорректные данные
5. **Документируйте тесты**: Понятные docstrings для каждого теста

## Troubleshooting

### ImportError: attempted relative import
- Убедитесь, что запускаете тесты из корня проекта
- Проверьте наличие `__init__.py` файлов

### ModuleNotFoundError
- Установите все зависимости: `pip install -r requirements.txt`
- Проверьте PYTHONPATH

### Медленные тесты
- Используйте `--fast` для пропуска медленных тестов
- Оптимизируйте фикстуры (используйте session scope)

## Контакты

При возникновении проблем или вопросов создайте issue в репозитории проекта.

