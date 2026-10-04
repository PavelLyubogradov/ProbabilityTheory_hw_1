"""Infection classification using categorical Naive Bayes.

Problem 1.24: given the test result (Test) and the age group
(Age_Group), predict the infection status (Status: Infected /
Not_infected). sklearn.naive_bayes.CategoricalNB is used because both
features are categorical.

Usage:
    python disease_naive_bayes.py

The data is downloaded directly from Google Drive via the DATA_PATH
link (file dataset_diseases.csv, see DATA_FILE_ID); there is no need
to download the file separately and place it next to the script --
only an internet connection is required.

Module architecture:
    The computational functions (load_data, encode_features,
    split_data, train_model, compute_metrics,
    predict_all_combinations) have no side effects (they neither
    print nor plot) and return only data. All console output and
    plotting are factored out into separate functions (print_*,
    plot_*) so that the computational part can be reused (e.g. in
    tests or in another script) without unwanted text or plots
    appearing on screen.
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
    """Loads the dataset from a local file or a URL.

    Algorithm:
        1. Read the CSV file into a DataFrame with pandas.
           pandas.read_csv can read a local disk path and a direct
           download link (here, a drive.google.com/uc?id=... URL)
           equally well: if path starts with http(s)://, pandas
           downloads the content itself and parses it as CSV, with
           no separate file-download step required.
        2. If reading fails (no network connection, the link is
           unreachable, the file is corrupted, or the server
           returned an HTML page instead of CSV -- e.g. a Google
           sign-in page), catch the error and re-raise it with a
           clear, human-readable message instead of a low-level
           pandas/urllib traceback.

    Args:
        path: Path or URL to the CSV file with the data. By default
            this is a direct Google Drive download link (DATA_PATH),
            but it can also be a local path to an already downloaded
            dataset_diseases.csv file.

    Returns:
        A table with the raw (not yet encoded) data.

    Raises:
        RuntimeError: If the file could not be downloaded or parsed
            as a valid CSV.
    """
    try:
        return pd.read_csv(path)
    except Exception as error:  # noqa: BLE001 - any read error
        raise RuntimeError(
            f"Failed to load data from {path!r}. Please check your "
            "internet connection, the accessibility of the link "
            "(the file on Google Drive must be shared with "
            "\"Anyone with the link\"), and the file format "
            f"(CSV is expected). Original error: {error}"
        ) from error


def print_data_summary(data: pd.DataFrame) -> None:
    """Prints a brief summary of the loaded dataset.

    Args:
        data: Table with the raw data (the result of load_data).

    Returns:
        None. The summary is printed to the console.
    """
    print("Dataset shape:", data.shape)
    print("\nNumber of objects per class:")
    print(data[TARGET_COLUMN].value_counts())
    print("\nFirst rows of the dataset:")
    print(data.head())


def encode_features(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, OrdinalEncoder, OrdinalEncoder]:
    """Encodes the categorical features and the target as integers.

    CategoricalNB requires integer-encoded categories rather than
    strings, so the features and the target are encoded with
    separate OrdinalEncoder instances, so predictions can later be
    decoded back into text labels.

    Algorithm:
        1. Create a separate OrdinalEncoder for the features (Test,
           Age_Group) and fit it with fit_transform on the
           FEATURE_COLUMNS columns -- the encoder finds the unique
           categories in each column itself and assigns them integer
           codes (0, 1, ...). These codes are arbitrary and carry no
           ordering (CategoricalNB uses them purely as category
           labels, not as numbers with a "greater/less" meaning).
        2. Create a second, independent OrdinalEncoder for the
           target (Status) and fit it on the TARGET_COLUMN column;
           .ravel() turns the result from a matrix of shape (n, 1)
           into a flat vector of shape (n,), as sklearn expects for
           y.
        3. The two encoders are kept separate on purpose: the
           feature encoder and the target encoder encode
           semantically different data, and target_encoder will
           later be needed on its own to decode predicted classes
           back into strings ("Infected"/"Not_infected").

    Args:
        data: The raw table with the FEATURE_COLUMNS features and
            the TARGET_COLUMN target as string categories.

    Returns:
        A tuple of four elements:
            encoded_features: The feature matrix after encoding.
            encoded_target: The encoded target vector.
            feature_encoder: The fitted OrdinalEncoder for the
                features, storing the original categories in
                ``categories_``.
            target_encoder: The fitted OrdinalEncoder for the
                target, needed to decode labels back later.
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
    """Prints the categories discovered by the encoders during fitting.

    Useful as a sanity check that encoding went as expected -- for
    instance, that the data contains no typos such as
    "young"/"Young", which the encoder would treat as different
    categories.

    Args:
        feature_encoder: Encoder fitted on the FEATURE_COLUMNS columns.
        target_encoder: Encoder fitted on the TARGET_COLUMN column.

    Returns:
        None. The summary is printed to the console.
    """
    print("\nCategories of the Test feature:", feature_encoder.categories_[0])
    print("Categories of the Age_Group feature:", feature_encoder.categories_[1])
    print("Categories of the target variable:", target_encoder.categories_[0])


def split_data(
    features: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits the data into training and test sets.

    Uses stratification on the target (stratify=target), so that
    the proportion of Infected/Not_infected is the same in the
    training and test sets.

    Algorithm:
        1. Call train_test_split from sklearn, passing the encoded
           features and target.
        2. Set test_size=TEST_SIZE (0.25) -- a quarter of the data
           goes to the test set, three quarters remain for training.
        3. Set stratify=target -- train_test_split first groups the
           objects by the value of the target and then draws
           test_size from each group separately. This guarantees
           that the Infected/Not_infected ratio in train and test
           (approximately) matches the ratio in the full dataset,
           rather than being left to chance.
        4. Fix random_state=RANDOM_STATE -- makes the split
           reproducible: running the script again with the same
           value produces the same train/test split.

    Args:
        features: Matrix of encoded features.
        target: Vector of the encoded target.

    Returns:
        A tuple ``(x_train, x_test, y_train, y_test)`` -- the
        training and test parts of the features and the target.
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
    """Trains CategoricalNB with Laplace smoothing alpha=ALPHA.

    Algorithm:
        1. Create a CategoricalNB object with the alpha=ALPHA
           parameter. Alpha is the Laplace smoothing parameter: it
           adds a small "virtual" count to the occurrence counts of
           every (feature, value, class) combination, so that
           combinations absent from the training set do not receive
           a zero (and thus product-nullifying) probability when
           classifying new objects.
        2. Call model.fit(x_train, y_train). Internally, sklearn
           computes, for every feature and every class, the
           conditional probabilities P(feature value | class) from
           the occurrence frequencies in the training data, as well
           as the prior class probabilities P(class) = the
           proportion of objects of that class in train. These
           probabilities are then multiplied together (under the
           "naive" assumption of feature independence given the
           class) to obtain predictions.
        3. Return the fitted model object, ready for calls to
           predict and predict_proba.

    Args:
        x_train: Encoded features of the training set.
        y_train: Encoded target of the training set.

    Returns:
        The fitted CategoricalNB model.
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
    """Computes quality metrics for the model on the test set.

    A pure function with no side effects: it only computes and
    returns values, it neither prints nor plots -- this allows the
    function to be reused (e.g. to compare several models' metrics
    or log them to a file) without unwanted output.

    Algorithm:
        1. Obtain the model's predictions on the test set:
           y_pred = model.predict(x_test). The model compares the
           posterior class probabilities for each object and picks
           the class with the highest probability.
        2. Determine the integer code of the POSITIVE_CLASS_LABEL
           class ("Infected") via target_encoder.categories_ -- this
           is needed to explicitly tell sklearn which class to treat
           as "positive" when computing precision/recall/F1
           (otherwise these metrics would default to the other
           class).
        3. Compute accuracy_score -- the fraction of correctly
           classified objects among all test objects.
        4. Compute precision_score -- the fraction of true Infected
           cases among all objects the model predicted as Infected
           (i.e. how much the model's positive predictions can be
           trusted).
        5. Compute recall_score -- the fraction of true Infected
           cases the model actually found (i.e. how many real
           infection cases were not missed).
        6. Compute f1_score -- the harmonic mean of precision and
           recall, a balanced metric useful when the class ratio is
           not perfectly even.
        7. Pack all the numeric metrics and the y_pred prediction
           vector into one dictionary and return it.

    Args:
        model: The fitted CategoricalNB model.
        x_test: Encoded features of the test set.
        y_test: Encoded target of the test set.
        target_encoder: The encoder used for the target; needed to
            obtain the class names.

    Returns:
        A dictionary with the keys "accuracy", "precision", "recall",
        "f1" (numbers) and "y_pred" (array of predicted classes).
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
    """Prints the quality metrics and the full classification_report.

    Args:
        metrics: The dictionary returned by compute_metrics.
        y_test: Encoded target of the test set.
        class_names: Class names in the order of their integer codes.

    Returns:
        None. The report is printed to the console.
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
    """Builds and displays the confusion matrix.

    Args:
        y_test: Encoded target of the test set.
        y_pred: Predicted classes from the model (from compute_metrics).
        class_names: Class names in the order of their integer codes.

    Returns:
        None. Draws a plot with matplotlib.
    """
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_names,
    )
    plt.title("CategoricalNB: confusion matrix (diseases)")
    plt.show()


def predict_all_combinations(
    model: CategoricalNB,
    feature_encoder: OrdinalEncoder,
    target_encoder: OrdinalEncoder,
) -> pd.DataFrame:
    """Builds predictions and probabilities for every feature combination.

    Enumerates all combinations of the Test and Age_Group categories
    (similar to the Weather Play example from the course template)
    and shows the posterior probabilities and the final class the
    model assigns to each of them.

    Algorithm:
        1. Take, from feature_encoder.categories_, the lists of all
           original categories for Test (Positive/Negative) and
           Age_Group (Young/Old) -- exactly the categories the
           encoder "saw" during fitting.
        2. Build a DataFrame all_cases with every possible pair
           (Test, Age_Group) via a Cartesian product (a nested loop:
           for each value of test -- for each value of age) --
           giving 2 x 2 = 4 rows, one per feature combination.
        3. Determine the integer codes of the "positive" and
           "negative" classes (Infected / Not_infected) via
           target_encoder, in order to label the probability columns
           correctly. The function is designed for exactly two
           classes.
        4. Encode all_cases with the same feature_encoder used
           during training (transform only, no re-fitting) -- this
           guarantees that strings such as "Positive", "Young", etc.
           turn into the same integer codes used when the model was
           trained.
        5. Obtain predict_proba from the model -- for each of the 4
           combinations this is a vector of two numbers: the
           posterior probability of the Not_infected class and of
           the Infected class (they sum to 1 for each row).
        6. Obtain predict -- the final class the model would choose
           for each combination (the class with the highest
           probability from step 5).
        7. Assemble the result into a table: the original feature
           categories, columns with the P(Not_infected) and
           P(Infected) probabilities, and a column with the
           prediction decoded back to text (via
           target_encoder.inverse_transform).

    Args:
        model: The fitted CategoricalNB model.
        feature_encoder: The encoder used for the features; its
            ``categories_`` give all possible values.
        target_encoder: The encoder used for the target; needed for
            the class names and to decode predictions back to text
            labels.

    Returns:
        A table with every feature combination, the probability of
        each class, and the model's final prediction.

    Raises:
        ValueError: If the target has a number of classes other than
            two (the function is designed for the binary case only).
    """
    class_names = list(target_encoder.categories_[0])
    if len(class_names) != 2:
        raise ValueError(
            "predict_all_combinations only supports a binary target, "
            f"but found classes: {class_names}"
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
    """Runs the full pipeline: data -> training -> evaluation.

    Algorithm:
        1. load_data -- read the CSV (with a clear error on failure),
           print_data_summary -- print the data summary.
        2. encode_features -- encode the categorical features and the
           target as integers suitable for CategoricalNB;
           print_encoding_summary -- show the discovered categories.
        3. split_data -- split the encoded data into training and
           test sets, stratified by class.
        4. train_model -- fit CategoricalNB on the training set.
        5. compute_metrics -- compute accuracy/precision/recall/F1 on
           the test set; print_metrics_report and
           plot_confusion_matrix -- display the results.
        6. predict_all_combinations -- for illustration, show the
           model's probabilities and prediction for every possible
           feature combination, not just the test objects.

    Returns:
        None. All results are printed to the console and plotted.
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
    print("\nTrain set size:", x_train.shape)
    print("Test set size:", x_test.shape)

    model = train_model(x_train, y_train)

    class_names = list(target_encoder.categories_[0])
    metrics = compute_metrics(model, x_test, y_test, target_encoder)
    print_metrics_report(metrics, y_test, class_names)
    plot_confusion_matrix(y_test, metrics["y_pred"], class_names)

    combinations = predict_all_combinations(
        model, feature_encoder, target_encoder
    )
    print("\nPredictions for every feature combination:")
    print(combinations.round(3))

if __name__ == "__main__":
    main()
