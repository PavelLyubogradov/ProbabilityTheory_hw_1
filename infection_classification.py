"""Классификация заражения с помощью категориального Наивного Байеса.

Задача 1.24: по признакам результата теста (Test) и возрастной группы
(Age_Group) предсказать статус заражения (Status: Infected /
Not_infected). Используется sklearn.naive_bayes.CategoricalNB, так как
оба признака являются категориальными.

Запуск:
    python disease_naive_bayes.py

Данные загружаются напрямую с Google Drive по ссылке DATA_PATH
(файл dataset_diseases.csv, см. DATA_FILE_ID); отдельно скачивать
файл и класть его рядом со скриптом не требуется, нужен только
доступ в интернет. 

Архитектура модуля:
    Вычислительные функции (load_data, encode_features, split_data,
    train_model, compute_metrics, predict_all_combinations) не имеют
    побочных эффектов (не печатают и не рисуют) и возвращают только
    данные. Весь вывод в консоль и построение графиков вынесены в
    отдельные функции (print_*, plot_*), чтобы вычислительную часть
    можно было переиспользовать (например, в тестах или в другом
    скрипте) без лишнего текста и графиков на экране.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

DATA_FILE_ID = "1TUwsyfVjl15gsf3YsXEgyCgiJYBDfXCS"
DATA_PATH = f"https://drive.google.com/uc?id={DATA_FILE_ID}"
FEATURE_COLUMNS = ["Test", "Age_Group"]
TARGET_COLUMN = "Status"
POSITIVE_CLASS_LABEL = "Infected"
TEST_SIZE = 0.25
RANDOM_STATE = 42
ALPHA = 1.0


def load_data(path: str) -> pd.DataFrame:
    """Загружает данные из локального файла или по URL.

    Алгоритм:
        1. Прочитать CSV-файл в DataFrame с помощью pandas.
           pandas.read_csv одинаково умеет читать как локальный путь
           на диске, так и прямую ссылку на скачивание (в нашем
           случае — ссылку вида drive.google.com/uc?id=...): если
           path начинается с http(s)://, pandas сам скачивает
           содержимое и парсит его как CSV, не требуя отдельного
           шага загрузки файла на диск.
        2. Если чтение не удалось (нет сети, ссылка недоступна, файл
           повреждён или сервер вернул не CSV, а HTML-страницу —
           например, страницу входа в Google), перехватить ошибку и
           поднять её заново с понятным, человекочитаемым сообщением
           вместо низкоуровневой трассировки pandas/urllib.

    Args:
        path: Путь или URL к CSV-файлу с данными. По умолчанию это
            прямая ссылка на скачивание с Google Drive (DATA_PATH),
            но может быть и локальный путь к уже скачанному файлу
            dataset_diseases.csv.

    Returns:
        Таблица с исходными (ещё не закодированными) данными.

    Raises:
        RuntimeError: Если файл не удалось загрузить или прочитать
            как корректный CSV.
    """
    try:
        return pd.read_csv(path)
    except Exception as error:  # noqa: BLE001 - любая ошибка чтения
        raise RuntimeError(
            f"Не удалось загрузить данные из {path!r}. Проверьте "
            "подключение к интернету, доступность ссылки (файл на "
            "Google Drive должен быть открыт по ссылке на просмотр) "
            "и формат файла (ожидается CSV). "
            f"Исходная ошибка: {error}"
        ) from error


def print_data_summary(data: pd.DataFrame) -> None:
    """Печатает краткую сводку по загруженному набору данных.

    Args:
        data: Таблица с исходными данными (результат load_data).

    Returns:
        None. Сводка выводится в консоль.
    """
    print("Размер датасета:", data.shape)
    print("\nКоличество объектов по классам:")
    print(data[TARGET_COLUMN].value_counts())
    print("\nПервые строки датасета:")
    print(data.head())


def encode_features(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, OrdinalEncoder, OrdinalEncoder]:
    """Кодирует категориальные признаки и целевую переменную в числа.

    CategoricalNB требует целочисленного кодирования категорий, а не
    строк, поэтому признаки и целевая переменная кодируются отдельными
    OrdinalEncoder, чтобы потом можно было декодировать предсказания
    обратно в текстовые метки.

    Алгоритм:
        1. Создать отдельный OrdinalEncoder для признаков (Test,
           Age_Group) и обучить его методом fit_transform на столбцах
           FEATURE_COLUMNS — энкодер сам находит уникальные категории
           по каждому столбцу и присваивает им целые коды (0, 1, ...).
           Эти коды произвольны и не означают порядок (CategoricalNB
           использует их только как метки категорий, а не как числа
           со смыслом "больше/меньше").
        2. Создать второй, независимый OrdinalEncoder для целевой
           переменной (Status) и обучить его на столбце
           TARGET_COLUMN; .ravel() превращает результат из матрицы
           формы (n, 1) в плоский вектор формы (n,), как ожидает
           sklearn для y.
        3. Два энкодера держатся раздельно умышленно: энкодер
           признаков и энкодер цели кодируют разные по смыслу данные,
           и впоследствии target_encoder понадобится отдельно, чтобы
           декодировать предсказанные классы обратно в строки
           ("Infected"/"Not_infected").

    Args:
        data: Исходная таблица с признаками FEATURE_COLUMNS и целевой
            переменной TARGET_COLUMN в виде строковых категорий.

    Returns:
        Кортеж из четырёх элементов:
            encoded_features: Матрица признаков после кодирования.
            encoded_target: Вектор закодированной целевой переменной.
            feature_encoder: Обученный OrdinalEncoder для признаков,
                хранит исходные категории в ``categories_``.
            target_encoder: Обученный OrdinalEncoder для целевой
                переменной, нужен для обратного декодирования меток.
    """
    feature_encoder = OrdinalEncoder()
    encoded_features = feature_encoder.fit_transform(
        data[FEATURE_COLUMNS]
    )

    target_encoder = OrdinalEncoder()
    encoded_target = target_encoder.fit_transform(
        data[[TARGET_COLUMN]]
    ).ravel()

    return encoded_features, encoded_target, feature_encoder, target_encoder


def print_encoding_summary(
    feature_encoder: OrdinalEncoder,
    target_encoder: OrdinalEncoder,
) -> None:
    """Печатает категории, обнаруженные энкодерами при обучении.

    Полезно как проверка того, что кодирование прошло ожидаемо —
    например, что в данных нет опечаток вроде "young"/"Young",
    которые энкодер воспримет как разные категории.

    Args:
        feature_encoder: Энкодер, обученный на столбцах FEATURE_COLUMNS.
        target_encoder: Энкодер, обученный на столбце TARGET_COLUMN.

    Returns:
        None. Сводка выводится в консоль.
    """
    print("\nКатегории признака Test:", feature_encoder.categories_[0])
    print("Категории признака Age_Group:", feature_encoder.categories_[1])
    print("Категории целевой переменной:", target_encoder.categories_[0])


def split_data(
    features: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Разбивает данные на обучающую и тестовую выборки.

    Использует стратификацию по целевой переменной (stratify=target),
    чтобы доля классов Infected/Not_infected была одинаковой в train
    и test.

    Алгоритм:
        1. Вызвать train_test_split из sklearn, передав закодированные
           признаки и целевую переменную.
        2. Указать test_size=TEST_SIZE (0.25) — четверть данных уходит
           в тестовую выборку, три четверти остаются для обучения.
        3. Указать stratify=target — train_test_split сначала
           группирует объекты по значению целевой переменной, а затем
           отбирает test_size от каждой группы отдельно. Это
           гарантирует, что соотношение Infected/Not_infected в train
           и test совпадает (приблизительно) с соотношением в полном
           датасете, а не определяется случайно.
        4. Зафиксировать random_state=RANDOM_STATE — делает разбиение
           воспроизводимым: при повторном запуске с тем же значением
           получится то же самое разбиение на train/test.

    Args:
        features: Матрица закодированных признаков.
        target: Вектор закодированной целевой переменной.

    Returns:
        Кортеж ``(x_train, x_test, y_train, y_test)`` — обучающая и
        тестовая части признаков и целевой переменной.
    """
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> CategoricalNB:
    """Обучает CategoricalNB со сглаживанием Лапласа alpha=ALPHA.

    Алгоритм:
        1. Создать объект CategoricalNB с параметром alpha=ALPHA.
           Alpha — параметр сглаживания Лапласа: он добавляет
           небольшую "виртуальную" добавку к счётчикам встречаемости
           каждой комбинации (признак, значение, класс), чтобы
           комбинации, которых не было в обучающей выборке, не
           получали нулевую (и тем самым "обнуляющую" всё
           произведение) вероятность при классификации новых
           объектов.
        2. Вызвать model.fit(x_train, y_train). Внутри sklearn для
           каждого признака и каждого класса считает условные
           вероятности P(значение_признака | класс) на основе частот
           встречаемости в обучающих данных, а также априорные
           вероятности классов P(класс) = доля объектов этого класса
           в train. Именно эти вероятности затем перемножаются (по
           "наивному" предположению о независимости признаков при
           фиксированном классе) для получения предсказаний.
        3. Вернуть обученный объект model, готовый к вызовам predict
           и predict_proba.

    Args:
        x_train: Закодированные признаки обучающей выборки.
        y_train: Закодированная целевая переменная обучающей выборки.

    Returns:
        Обученная модель CategoricalNB.
    """
    model = CategoricalNB(alpha=ALPHA)
    model.fit(x_train, y_train)
    return model


def compute_metrics(
    model: CategoricalNB,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_encoder: OrdinalEncoder,
) -> dict[str, float | np.ndarray]:
    """Вычисляет метрики качества модели на тестовой выборке.

    Чистая функция без побочных эффектов: только считает и
    возвращает значения, не печатает и не рисует графики — это
    позволяет переиспользовать её (например, сравнивать метрики
    нескольких моделей или логировать их в файл) без лишнего вывода.

    Алгоритм:
        1. Получить предсказания модели на тестовой выборке:
           y_pred = model.predict(x_test). Модель сравнивает
           апостериорные вероятности классов для каждого объекта и
           выбирает класс с максимальной вероятностью.
        2. Определить числовой код класса POSITIVE_CLASS_LABEL
           ("Infected") через target_encoder.categories_ — нужен,
           чтобы явно указать sklearn, какой класс считать
           "положительным" при расчёте precision/recall/F1 (иначе
           эти метрики по умолчанию считались бы для другого класса).
        3. Посчитать accuracy_score — долю верно классифицированных
           объектов среди всех объектов теста.
        4. Посчитать precision_score — долю истинных Infected среди
           всех объектов, предсказанных моделью как Infected (то
           есть насколько можно доверять положительному предсказанию
           модели).
        5. Посчитать recall_score — долю истинных Infected, которых
           модель действительно нашла (то есть сколько реальных
           случаев заражения не было пропущено).
        6. Посчитать f1_score — гармоническое среднее precision и
           recall, сбалансированная метрика, полезная при не совсем
           равном соотношении классов.
        7. Сложить все числовые метрики и вектор предсказаний y_pred
           в один словарь и вернуть его.

    Args:
        model: Обученная модель CategoricalNB.
        x_test: Закодированные признаки тестовой выборки.
        y_test: Закодированная целевая переменная тестовой выборки.
        target_encoder: Encoder, использованный для целевой
            переменной; нужен, чтобы получить имена классов.

    Returns:
        Словарь с ключами "accuracy", "precision", "recall",
        "f1" (числа) и "y_pred" (массив предсказанных классов).
    """
    y_pred = model.predict(x_test)

    class_names = list(target_encoder.categories_[0])
    positive_code = class_names.index(POSITIVE_CLASS_LABEL)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, pos_label=positive_code
        ),
        "recall": recall_score(y_test, y_pred, pos_label=positive_code),
        "f1": f1_score(y_test, y_pred, pos_label=positive_code),
        "y_pred": y_pred,
    }


def print_metrics_report(
    metrics: dict[str, float | np.ndarray],
    y_test: np.ndarray,
    class_names: list[str],
) -> None:
    """Печатает метрики качества и полный classification_report.

    Args:
        metrics: Словарь, возвращённый compute_metrics.
        y_test: Закодированная целевая переменная тестовой выборки.
        class_names: Имена классов в порядке их числовых кодов.

    Returns:
        None. Отчёт выводится в консоль.
    """
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")

    print(
        "\n"
        + classification_report(
            y_test,
            metrics["y_pred"],
            target_names=class_names,
            digits=3,
        )
    )


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    """Строит и показывает матрицу ошибок.

    Args:
        y_test: Закодированная целевая переменная тестовой выборки.
        y_pred: Предсказанные моделью классы (из compute_metrics).
        class_names: Имена классов в порядке их числовых кодов.

    Returns:
        None. Строит график через matplotlib.
    """
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_names,
    )
    plt.title("CategoricalNB: матрица ошибок (diseases)")
    plt.show()


def predict_all_combinations(
    model: CategoricalNB,
    feature_encoder: OrdinalEncoder,
    target_encoder: OrdinalEncoder,
) -> pd.DataFrame:
    """Строит прогноз и вероятности для всех комбинаций признаков.

    Перебирает все сочетания категорий Test и Age_Group (аналогично
    примеру Weather Play из шаблона курса) и показывает, какие
    апостериорные вероятности и итоговый класс даёт модель для
    каждого из них.

    Алгоритм:
        1. Взять из feature_encoder.categories_ списки всех исходных
           категорий для Test (Positive/Negative) и Age_Group
           (Young/Old) — именно те категории, которые энкодер
           "увидел" при обучении.
        2. Сформировать DataFrame all_cases со всеми возможными
           парами (Test, Age_Group) через декартово произведение
           (вложенный перебор: для каждого значения test — для
           каждого значения age) — получится 2 × 2 = 4 строки, по
           одной на каждую комбинацию признаков.
        3. Определить числовые коды "положительного" и
           "отрицательного" классов (Infected / Not_infected) через
           target_encoder, чтобы корректно подписать столбцы с
           вероятностями. Функция рассчитана ровно на два класса.
        4. Закодировать all_cases тем же feature_encoder, что
           использовался при обучении (только transform, без
           повторного fit) — это гарантирует, что строки "Positive",
           "Young" и т.д. превратятся в те же числовые коды, что и
           при обучении модели.
        5. Получить от модели predict_proba — для каждой из 4
           комбинаций это вектор из двух чисел: апостериорная
           вероятность класса Not_infected и класса Infected (они в
           сумме дают 1 для каждой строки).
        6. Получить predict — итоговый класс, который модель выбрала
           бы для каждой комбинации (класс с максимальной
           вероятностью из шага 5).
        7. Собрать результат в таблицу: исходные категории признаков,
           столбцы с вероятностями P(Not_infected) и P(Infected), и
           столбец с декодированным обратно в текст предсказанием
           (через target_encoder.inverse_transform).

    Args:
        model: Обученная модель CategoricalNB.
        feature_encoder: Encoder, использованный для признаков; из
            его ``categories_`` берутся все возможные значения.
        target_encoder: Encoder, использованный для целевой
            переменной; нужен для имён классов и декодирования
            предсказаний обратно в текстовые метки.

    Returns:
        Таблица со всеми комбинациями признаков, вероятностями по
        каждому классу и итоговым прогнозом модели.

    Raises:
        ValueError: Если целевая переменная содержит не ровно два
            класса (функция рассчитана только на бинарный случай).
    """
    class_names = list(target_encoder.categories_[0])
    if len(class_names) != 2:
        raise ValueError(
            "predict_all_combinations поддерживает только бинарную "
            f"целевую переменную, но найдено классов: {class_names}"
        )

    test_values = feature_encoder.categories_[0]
    age_values = feature_encoder.categories_[1]

    all_cases = pd.DataFrame(
        [(test, age) for test in test_values for age in age_values],
        columns=FEATURE_COLUMNS,
    )

    positive_code = class_names.index(POSITIVE_CLASS_LABEL)
    negative_code = 1 - positive_code

    encoded_cases = feature_encoder.transform(all_cases)
    probabilities = model.predict_proba(encoded_cases)
    predictions = model.predict(encoded_cases)

    result = all_cases.copy()
    result[f"P({class_names[negative_code]})"] = probabilities[
        :, negative_code
    ]
    result[f"P({class_names[positive_code]})"] = probabilities[
        :, positive_code
    ]
    result["prediction"] = target_encoder.inverse_transform(
        predictions.reshape(-1, 1)
    ).ravel()

    return result


def main() -> None:
    """Запускает полный поцесс: данные -> обучение -> оценка.

    Алгоритм:
        1. load_data — прочитать CSV (с понятной ошибкой при сбое
           загрузки), print_data_summary — вывести сводку по данным.
        2. encode_features — закодировать категориальные признаки и
           целевую переменную в числовой вид, пригодный для
           CategoricalNB; print_encoding_summary — показать найденные
           категории.
        3. split_data — разбить закодированные данные на обучающую и
           тестовую выборки со стратификацией по классу.
        4. train_model — обучить CategoricalNB на обучающей выборке.
        5. compute_metrics — вычислить accuracy/precision/recall/F1 на
           тестовой выборке; print_metrics_report и
           plot_confusion_matrix — показать результаты.
        6. predict_all_combinations — для наглядности показать
           вероятности и прогноз модели по всем возможным сочетаниям
           признаков, а не только по объектам теста.

    Returns:
        None. Все результаты выводятся в консоль и в виде графиков.
    """
    try:
        data = load_data(DATA_PATH)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    print_data_summary(data)

    features, target, feature_encoder, target_encoder = encode_features(
        data
    )
    print_encoding_summary(feature_encoder, target_encoder)

    x_train, x_test, y_train, y_test = split_data(features, target)
    print("\nРазмер train:", x_train.shape)
    print("Размер test:", x_test.shape)

    model = train_model(x_train, y_train)

    class_names = list(target_encoder.categories_[0])
    metrics = compute_metrics(model, x_test, y_test, target_encoder)
    print_metrics_report(metrics, y_test, class_names)
    plot_confusion_matrix(y_test, metrics["y_pred"], class_names)

    combinations = predict_all_combinations(
        model, feature_encoder, target_encoder
    )
    print("\nПрогноз для всех комбинаций признаков:")
    print(combinations.round(3))


if __name__ == "__main__":
    main()
