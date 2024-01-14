# Data manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Progress bar
from tqdm import tqdm

# Signal processing
from scipy.signal import cwt, ricker
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# Machine learning
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (
    precision_score, recall_score, roc_curve, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, auc, precision_recall_curve
)
from sklearn.metrics import make_scorer

# Parallel processing
from joblib import Parallel, delayed

def import_data(n):
    """
    Imports and processes EEG data from multiple CSV files, combining specified
    time points for each sample.

    This function reads EEG data, processes it by grouping trials, filtering based
    on the number of time points (n) to combine, sorting the data, and assigning new
    sequential 'sample' values for each group of combined time points. Each 'sample'
    in the processed data represents a combined duration from the original dataset.

    Parameters:
    - n: int
        The number of time points to combine for processing the EEG data. Each time
        point represents 1.5 seconds. The value of `n` should be divisible by 3072
        to ensure equal chunk sizes.

    Returns:
    - pandas.DataFrame
        The processed EEG data, with trials aggregated and transformed according to
        the specified number of time points.

    Raises:
    - ValueError: If `n` is not divisible by 3072, which is necessary to ensure
      equal chunk sizes in the processed data.

    Examples:
    >>> processed_eeg_data = import_data(16)
    """

    if 3072 % n != 0:
        raise ValueError("The parameter 'n' should be divisible by 3072 to ensure equal chunk sizes.")

    master_df = pd.DataFrame()
    column_list = pd.read_csv("./dataset/columnLabels.csv").columns
    demographic = pd.read_csv("./dataset/demographic.csv")
    diagnosis_dict = dict(zip(demographic.subject, demographic[" group"]))

    for person_number in tqdm(range(1, 81 + 1)):
        csv_path = f"./dataset/{person_number}.csv/{person_number}.csv"

        df_processed = (pd.read_csv(csv_path, header=None, names=column_list)
                        .groupby("trial")
                        .filter(lambda x: len(x) == 9216)
                        .sort_values(by=['condition', 'trial', 'sample'])
                        .reset_index(drop=True)
                        .groupby(lambda x: x // n)
                        .mean()
                        .assign(sample=lambda x: list(range(1, int(9216 / n) + 1)) * x.trial.nunique())
                        .astype({"subject": 'uint8', "trial": 'uint8', "condition": 'uint8', "sample": 'uint16'}))

        master_df = pd.concat([master_df, df_processed], ignore_index=True)

    master_df['diagnosis'] = master_df.subject.map(diagnosis_dict)
    return master_df


def extract_features(df, scale_range=(1, 32, 2), n_jobs=-1, return_ml_ready=False):
    """
   Extracts and combines Continuous Wavelet Transform (CWT) and Fast Fourier Transform (FFT)
   features from EEG data represented in a given DataFrame.

   This custom-designed function is tailored for the project's EEG dataset. It processes each row of the
   input DataFrame to compute CWT and FFT features.
   The function reshapes the DataFrame, extracts features using CWT and FFT, and combines these
   features into a single DataFrame. Additionally, it maintains a mapping of subjects to diagnoses.
   Optionally, it can return the data in a format ready for machine learning models.

   Parameters:
   - df: pandas.DataFrame
       The input DataFrame containing EEG data. It should include columns for 'subject',
       'trial', 'condition', 'samples (time points)' and EEG measurement columns.
   - scale_range: tuple of three integers (start, stop, step), default=(1, 32, 2)
       The range of scales to be used for the CWT feature extraction, tailored to the
       characteristics of EEG signals.
  - n_jobs: int, default=-1
       The number of jobs to run in parallel during the feature extraction process.
       `-1` means using all processors.
   - return_ml_ready: bool, default=False
       If True, the function returns machine learning-ready data (features `X`,
       target `y`, and train-test split). If False, returns the DataFrame with
       CWT and FFT transformed features.

   Returns:
   - If return_ml_ready is False:
       final_df: pandas.DataFrame
           A DataFrame containing the extracted features combined with the
           subject diagnosis mapping. The DataFrame includes CWT and FFT features
           for each subject, tailored for EEG data analysis, along with the diagnosis.
   - If return_ml_ready is True:
       X: numpy.ndarray
           The feature array ready for machine learning models.
       y: numpy.ndarray
           The target array.
       (X_train, X_test, y_train, y_test): tuple
           The train-test split of the data.

   Notes:
   - The function is specifically designed for EEG data and expects the input DataFrame to have
     specific columns ('subject', 'trial', 'condition', 'samples') for proper functioning.
   - The computation of CWT and FFT features is parallelized to utilize available CPU resources,
     controlled by the `n_jobs` parameter.

   Examples:
   >>> import pandas as pd
   >>> eeg_data = {'subject': [1, 1, 2, 2], 'trial': [1, 2, 1, 2], 'condition': [1, 2, 2, 3],
                   'measurement1': [0.5, 0.6, 0.7, 0.8], 'measurement2': [1.5, 1.6, 1.7, 1.8],
                   'diagnosis': ['Healthy', 'Healthy', 'Disease', 'Disease']}
   >>> df = pd.DataFrame(eeg_data)
   >>> features_df = extract_features(df)
   >>> print(features_df.head())

   >>> X, y, (X_train, X_test, y_train, y_test) = extract_features(df, return_ml_ready=True)
   >>> print(X.shape, y.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
   """

    # Create a mapping for diagnosis and subject


    diagnosis_mapping = df[['subject', 'diagnosis']].drop_duplicates()

    # Reshape the DataFrame
    reshaped = df[df.columns.difference(['sample', 'diagnosis'])].groupby(["subject", "trial", 'condition']).apply(
        lambda x: x.iloc[:, :-3].T.values.reshape(-1))

    reshaped_df = reshaped.reset_index()
    reshaped_df.columns = ['subject', 'trial', 'condition', 'features']

    # Define the scales for CWT using the provided scale range
    scales = np.arange(scale_range[0], scale_range[1], scale_range[2])

    # Generate column names for CWT features
    cwt_feature_names = []
    for scale in scales:
        cwt_feature_names.extend([f'cwt_{scale}_mean', f'cwt_{scale}_std', f'cwt_{scale}_median',
                                  f'cwt_{scale}_max',
                                  f'cwt_{scale}_min',
                                  f'cwt_{scale}_skew', f'cwt_{scale}_kurt'])
    fft_feature_names = ['fft_mean', 'fft_std', 'fft_median',
                         'fft_max', 'fft_min',
                         'fft_skew', 'fft_kurt']


    # Function to process each row
    def process_row(row):
        # Extract the feature part
        i = row['features']

        # Apply CWT
        cwtmatr = cwt(i, ricker, scales)

        # Extract CWT Features
        cwt_features = np.concatenate((
            np.mean(cwtmatr, axis=1),
            np.std(cwtmatr, axis=1),
            np.median(cwtmatr, axis=1),
            np.max(cwtmatr, axis=1),
            np.min(cwtmatr, axis=1),
            skew(cwtmatr, axis=1),
            kurtosis(cwtmatr, axis=1, fisher=True, bias=False)
        ))

        # Apply FFT
        fft_transform = np.abs(fft(i))

        # Extract FFT Features
        fft_features = np.array([
            np.mean(fft_transform),
            np.std(fft_transform),
            np.median(fft_transform),
            np.max(fft_transform),
            np.min(fft_transform),
            skew(fft_transform),
            kurtosis(fft_transform, fisher=True, bias=False)
        ])

        # Combine CWT and FFT features
        combined_features = np.concatenate((cwt_features, fft_features))
        return np.concatenate(([row['subject']], combined_features))


    # Process each row in parallel
    n_jobs = n_jobs
    results = Parallel(n_jobs=n_jobs)(delayed(process_row)(row) for row in tqdm(reshaped_df.to_dict('records')))

    # Convert the results to a DataFrame
    features_df = pd.DataFrame(results)

    # Define column names for the features
    column_names = ['subject'] + cwt_feature_names + fft_feature_names
    features_df.columns = column_names

    # Merge with diagnosis mapping
    final_df = pd.merge(features_df, diagnosis_mapping, on=["subject"])

    if return_ml_ready:
        X = np.vstack(final_df
                      .drop(["subject", "diagnosis"], axis=1)
                      .groupby(final_df.index // 3)
                      .apply(lambda x: x.values.flatten()))

        y = final_df.diagnosis.values[::3]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True)
        return X, y, (X_train, X_test, y_train, y_test)

    return final_df


def evaluate_model(model, X, y, cv_folds=5, n_jobs=-1):
    """
    Evaluate the performance of a classification model using cross-validation
    and various metrics.

    This function performs cross-validation to assess the model's performance
    in terms of accuracy, precision, recall, and ROC-AUC. It also evaluates
    the model's ability to predict probabilities by generating a confusion
    matrix, ROC curve, and Precision-Recall curve if applicable.

    Parameters:
    - model: A scikit-learn-compatible model object
        The classification model to be evaluated.
    - X: array-like of shape (n_samples, n_features)
        The input features for model evaluation.
    - y: array-like of shape (n_samples,)
        The target values (class labels) for model evaluation.
    - cv_folds: int, default=5
        The number of folds to use for cross-validation.
    - n_jobs: int, default=-1
        The number of jobs to run in parallel during cross-validation.
        `-1` means using all processors.

    Returns:
    - results: dict
        A dictionary containing evaluation metrics and, if applicable,
        additional evaluation elements such as confusion matrix, ROC curve,
        and Precision-Recall curve. The ROC curve and Precision-Recall curve
        are included only if the model supports probability predictions.
        Each metric includes the mean and standard deviation across CV folds.

    Raises:
    - ValueError: If `cv_folds` is less than 2, or if `X` and `y` have mismatched lengths.
    - TypeError: If `model` is not a scikit-learn-compatible model.

    Notes:
    - The function checks if the model supports probability predictions (via
      `predict_proba`). If not, relevant metrics are not calculated.
    - StratifiedKFold is used for cross-validation to maintain the percentage
      of samples for each class.
    - The evaluations for confusion matrix, ROC curve, and Precision-Recall
      curve are based on a separate train-test split and are not part of the
      cross-validation process.

    Examples:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = RandomForestClassifier()
    >>> results = evaluate_model(model, X, y, cv_folds=10, n_jobs=2)
    >>> print(results['accuracy'])
    {'mean': 0.96, 'std': 0.02}
    """

    if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
        raise TypeError("Provided model is not a valid scikit-learn-compatible model.")

    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    if len(X) != len(y):
        raise ValueError("The length of X and y must be the same.")

    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0, average='binary'),
        'recall': make_scorer(recall_score, zero_division=0, average='binary'),
        'roc_auc': 'roc_auc'
    }

    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=folds, scoring=scoring, n_jobs=n_jobs, verbose=10)

    results = {}
    for metric in scoring.keys():
        mean_score = cv_results[f'test_{metric}'].mean()
        std_score = cv_results[f'test_{metric}'].std()
        results[metric] = {'mean': mean_score, 'std': std_score}

    # Check if the model can predict probabilities
    if hasattr(model, "predict_proba"):
        # Separate train-test split for ROC and Precision-Recall curves calculation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        results["confusion_matrix"] = conf_matrix

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        roc_metrics = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc_score": auc_score}
        results["roc_curve"] = roc_metrics

        # Precision-Recall Curve and AUC score
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
        auc_score_pr = auc(recall, precision)  # Calculate AUC for Precision-Recall Curve
        pr_metrics = {"precision": precision, "recall": recall, "thresholds": thresholds_pr, "auc_score": auc_score_pr}
        results["precision_recall_curve"] = pr_metrics
    else:
        results["roc_curve"] = "Model does not support probability predictions"
        results["precision_recall_curve"] = "Model does not support probability predictions"

    return results


def plot_roc_curves(tree_based_models, mode='light', figsize=(8, 8), color_palette='tab10', dpi=100,ax=None,title=None):
    # Set the plotting style based on the specified mode
    plt.style.use("dark_background" if mode == 'dark' else "default")
    middle_line_color = 'lightgray' if mode == 'dark' else 'navy'
    title = 'ROC Curve' if title is None else title

    # Sort the tree-based models by their AUC score
    sorted_models = sorted(tree_based_models.items(), key=lambda x: x[1]['roc_curve']['auc_score'])

    # Get the colormap and create a normalization instance
    cmap = colormaps[color_palette]
    norm = plt.Normalize(0, len(sorted_models))

    # Create the plot with specified size and resolution
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Add a diagonal line representing random guessing
    ax.plot([0, 1], [0, 1], color=middle_line_color, linestyle='--', label="Random Guess")

    # Plot ROC curves for each model
    for i, (model_name, data) in enumerate(sorted_models):
        roc = data['roc_curve']
        ax.plot(roc['fpr'], roc['tpr'], label=f"{model_name} (AUC = {roc['auc_score']:.2f})", color=cmap(norm(i)))

    # Configure plot axes and title
    ax.set(xlim=[0, 1], ylim=[0, 1], xlabel='False Positive Rate', ylabel='True Positive Rate',
           title=title, aspect='equal')

    # Add a legend in the lower right corner
    ax.legend(loc="lower right")

    # Adjust the transparency of plot spines
    for spine in ax.spines.values():
        spine.set_alpha(0.4)

    # Return the figure object for further use
    return ax


def plot_precision_recall_curves(tree_based_models, mode='light', figsize=(6, 6), dpi=100, color_palette='tab10'):
    if mode == 'dark':
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    # Assuming sorted_d contains the data for different models
    sorted_d = dict(sorted(tree_based_models.items(), key=lambda x: x[1]['roc_curve']['auc_score']))

    # Define the colormap and normalize it
    cmap = colormaps[color_palette]
    norm = plt.Normalize(0, len(sorted_d))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plotting precision-recall curve for each model with different colors from the colormap
    for i, (model_name, model_results) in enumerate(sorted_d.items()):
        precision_recall = model_results['precision_recall_curve']
        auc_score = auc(precision_recall['recall'], precision_recall['precision'])
        label = f'{model_name} (AUC: {auc_score:.2f})'
        color = cmap(norm(i))
        ax.plot(precision_recall['recall'], precision_recall['precision'], label=label, color=color)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')

    for spine in ax.spines.values():
        spine.set_alpha(0.4)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_facecolor("None")
    ax.legend()

    return fig





def plot_confusion_matrix(cm, color_palette='Blues', mode='light', figsize=(4, 4)):
    plt.style.use('default')
    if mode == 'dark':
        plt.style.use('dark_background')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=figsize)

    disp.plot(cmap=color_palette, include_values=True, colorbar=False, ax=ax)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig('confusion_matrix.png', transparent=True)
    plt.show()


def format_model_metrics(model_results):
    """
    Formats the performance metrics of various models for easier analysis.

    This function sorts the provided models based on their mean accuracy,
    then formats their performance metrics into a readable string format.
    It expects each model to have a dictionary of metrics, with each metric
    containing a dictionary with 'mean' and 'std' (standard deviation) values.

    Parameters:
    model_results (dict): A dictionary where keys are model names and values
                          are dictionaries of performance metrics.

    Returns:
    pd.DataFrame: A pandas DataFrame where each row represents a model and
                  columns represent different metrics formatted as 'mean ± std'.
    """
    # Sort the models based on mean accuracy
    sorted_result = dict(sorted(model_results.items(), key=lambda x: x[1]['accuracy']['mean']))

    # Prepare data for the DataFrame
    data_for_df = {}
    for model, metrics in sorted_result.items():
        data_for_df[model] = {}
        for metric, values in metrics.items():
            # Check if values is a dictionary and contains 'mean' and 'std'
            if isinstance(values, dict) and 'mean' in values and 'std' in values:
                mean = values['mean']
                std = values['std']
                data_for_df[model][metric] = f'{mean * 100:.2f} ± {std * 100:.2f}'

    # Create and return the DataFrame
    df = pd.DataFrame(data_for_df).T
    return df