import os

import soundfile as sf
from sklearn.model_selection import train_test_split
import pandas as pd


def merge_ears_filepaths_with_metadata(paths_df: pd.DataFrame, meta_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Merge EARS file paths DataFrame with EARS person metadata DataFrame on 'person' column.

    Args:
        paths_df (pd.DataFrame): DataFrame containing file paths with 'person' column.
        meta_df (pd.DataFrame): DataFrame containing EARS person metadata with index as 'person'.

    Returns:
        pd.DataFrame: Merged DataFrame containing file paths and corresponding person metadata.
    """
    merged_df = pd.merge(paths_df, meta_df, left_on="person", right_index=True, how="inner")
    if verbose:
        print("Merged EARS Metadata Summary:")
        print(merged_df.head())
    return merged_df


def get_ears_personal_metadata(path: str) -> pd.DataFrame:
    """
    Load EARS person metadata from a JSON file.
    
    Args:
        path (str): Path to the JSON file containing EARS person metadata.

    Returns:
        pd.DataFrame: DataFrame containing the EARS person metadata.
    """
    meta_df = pd.read_json(path).T
    return meta_df


def preprocess_ears_metadata(path: str, fast: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Preprocess EARS dataset metadata. Adds columns for length, style, emotion, freeform speech, and non-speech.

    Args:
        path (str): Path to the EARS dataset.
        fast (bool, optional): If True, skip length calculation for speed. Defaults to True.
        verbose (bool, optional): If True, print summary statistics. Defaults to False.

    Returns:
        pd.DataFrame: Preprocessed EARS metadata DataFrame.
    """
    df = _create_ears_paths(path)

    if not fast:
        df["length [s]"] = df["path"].apply(_count_len)

    df["style"] = df["file"].apply(_categorize_style)
    df["emotion"] = df["file"].apply(_categorize_emotion)
    df["is_freeform_speech"] = df["file"].apply(_categorize_freeform_speech)
    df["is_non_speech"] = df["file"].apply(_categorize_non_speech)

    df["style"] = df["style"].astype("category")
    df["emotion"] = df["emotion"].astype("category")

    if verbose:
        print("EARS Metadata Preprocessing Summary:")
        print(df.head())
    return df


def preprocess_wham_metadata(
        *,
        wham_data_cv: str,
        wham_data_tt: str,
        wham_data_tr: str,
        wham_noise_cv: str,
        wham_noise_tt: str,
        wham_noise_tr: str,
        wham_files_cv: str,
        wham_files_tt: str,
        wham_files_tr: str,
        fast: bool = True,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Preprocess WHAM dataset metadata by merging data and noise CSV files and adding file paths.

    Args:
        wham_data_cv (str): Path to WHAM metadata cv CSV file.
        wham_data_tt (str): Path to WHAM metadata tt CSV file.
        wham_data_tr (str): Path to WHAM metadata tr CSV file.
        wham_noise_cv (str): Path to WHAM noise cv CSV file.
        wham_noise_tt (str): Path to WHAM noise tt CSV file.
        wham_noise_tr (str): Path to WHAM noise tr CSV file.
        wham_files_cv (str): Path to WHAM cv audio files directory.
        wham_files_tt (str): Path to WHAM tt audio files directory.
        wham_files_tr (str): Path to WHAM tr audio files directory.
        fast (bool, optional): If True, skip length calculation for speed. Defaults to True.
        verbose (bool, optional): If True, print summary statistics. Defaults to False.

    Returns:
        pd.DataFrame: Preprocessed WHAM metadata DataFrame.
    """
    wham_data_cv_df = pd.read_csv(wham_data_cv)
    wham_data_tt_df = pd.read_csv(wham_data_tt)
    wham_data_tr_df = pd.read_csv(wham_data_tr)

    wham_noise_cv_df = pd.read_csv(wham_noise_cv)
    wham_noise_tt_df = pd.read_csv(wham_noise_tt)
    wham_noise_tr_df = pd.read_csv(wham_noise_tr)

    wham_data_cv_df["path"] = wham_data_cv_df["utterance_id"].map(lambda f: os.path.join(wham_files_cv, f))
    wham_data_tt_df["path"] = wham_data_tt_df["utterance_id"].map(lambda f: os.path.join(wham_files_tt, f))
    wham_data_tr_df["path"] = wham_data_tr_df["utterance_id"].map(lambda f: os.path.join(wham_files_tr, f))

    wham_noise_df = pd.concat([wham_noise_tr_df, wham_noise_tt_df, wham_noise_cv_df], ignore_index=True)
    wham_data_df = pd.concat([wham_data_tr_df, wham_data_tt_df, wham_data_cv_df], ignore_index=True)
    wham_df = pd.merge(wham_noise_df, wham_data_df, on="utterance_id", how="inner")
    if not fast:
        wham_df["len"] = wham_df["path"].map(_count_len)

    if verbose:
        print("WHAM Metadata Preprocessing Summary:")
        print(wham_df.head())
    return wham_df


def prepare_for_training(df: pd.DataFrame, train_percentage: float, reduce_to: float | int | None, filter_to: dict[str, list[str]] | None = None, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare dataset by splitting into train, validate, and test sets. Also reduces the size of each set based on the reduce_to parameter.

    Args:
        df (pd.DataFrame): The input dataframe to split.
        train_percentage (float): The percentage of data to use for training set. Defaults to 0.8 (80%). Test and validation sets will each get half of the remaining data.
        reduce_to (float | int | None): If float in (0,1], reduces the dataset to that fraction. If int > 1, reduces the dataset to that many samples. If None, no reduction is applied.
        filter_to (dict[str, list[str]] | None): A dictionary where keys are column names and values are lists of acceptable values for those columns. If provided, filters the dataset before splitting. Defaults to None.
        verbose (bool): If True, prints the sizes of the resulting datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the train, validate, and test dataframes.
    """
    if filter_to is not None:
        df = _filter_dataset(df, filter_to)

    train_df, temp_df = train_test_split(df, train_size=train_percentage, random_state=42)
    validate_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df = _reduce_dataset(train_df, reduce_to)
    validate_df = _reduce_dataset(validate_df, reduce_to)
    test_df = _reduce_dataset(test_df, reduce_to)
    if verbose:
        print(f"Train set size: {len(train_df)}")
        print(f"Validate set size: {len(validate_df)}")
        print(f"Test set size: {len(test_df)}")
    return train_df, validate_df, test_df


# ==============================================================================
#                       CATEGORY: PREPROCESS METADATA
#                 Helper functions for preprocessing metadata
# ==============================================================================


# ------------------------------------------------------------------------------
# EARS Dataset - Helper Functions
# ------------------------------------------------------------------------------
    
def _create_ears_paths(ears_dataset_path: str) -> pd.DataFrame:
    """
    Create a DataFrame with path, file, person information to all .wav files in the EARS dataset.

    Args:
        ears_dataset_path (str): Path to the EARS dataset.

    Returns:
        pd.DataFrame: DataFrame containing 'person', 'file', and 'path' columns for each .wav file.
    """
    rows = []
    max_person_index = 107
    for i in range(max_person_index):
        person = "p" + f"{i+1}".zfill(3) + "_resampled"
        person_dir = os.path.join(ears_dataset_path, person)
        
        if not os.path.exists(person_dir):
            raise ValueError(f"Path does not exists: {person_dir}")
        
        for file in os.listdir(person_dir):
            if file.endswith(".wav"):
                rows.append(
                    {
                        "person": person.replace("_resampled", ""),
                        "file": file,
                        "path": os.path.join(person_dir, file)
                    }
                )
        
    return pd.DataFrame(rows)

def _categorize_style(filename: str) -> str:
    """
    Categorize the speaking style based on given information in the filename.

    Args:
        filename (str): The filename to categorize.

    Returns:
        str: The categorized style from given set or "other" if none match.
    """
    styles = {"regular", "loud", "whisper", "fast", "slow", "highpitch", "lowpitch"}
    return next((s for s in styles if s in filename), "other")

def _categorize_emotion(filename: str) -> str:
    """
    Categorize the emotion based on given information in the filename.

    Args:
        filename (str): The filename to categorize.

    Returns:
        str: The categorized emotion from given set or "other" if none match.
    """
    emotions = {'adoration', 'fear', 'pain', 'realization', 'confusion',
       'cuteness', 'distress', 'guilt', 'amazement', 'contentment',
       'pride', 'desire', 'relief', 'disappointment', 'disgust',
       'embarassment', 'anger', 'interest', 'serenity', 'sadness',
       'amusement', 'extasy', 'neutral'}
    return next((e for e in emotions if e in filename), "other")
    
def _categorize_freeform_speech(filename: str) -> bool:
    """
    Categorize if the filename corresponds to clean long freeform speech.

    Args:
        filename (str): The filename to categorize.

    Returns:
        bool: True if the filename corresponds to freeform speech, False otherwise.
    """
    return True if "freeform_speech" in filename else False

def _categorize_non_speech(filename: str) -> bool:
    """
    Categorize if the filename corresponds to non-speech sounds.
    
    Args:
        filename (str): The filename to categorize.
        
    Returns:
        bool: True if the filename corresponds to non-speech sounds, False otherwise.
    """
    non_speech = {'interjection_greetings', 'vegetative_throat',
       'nonverbal_screaming', 'nonverbal_crying', 'vegetative_eating',
       'melodic_happy_birthday', 'vegetative_yawning',
       'nonverbal_laughter_open', 'nonverbal_cheering',
       'nonverbal_yelling', 'vegetative_coughing',
       'nonverbal_laughter_closed', 'interjection_agreement',
       'interjection_filler', 'vegetative_sneezing',
       'interjection_congratulations'}
    return any(ns in filename for ns in non_speech)


# ------------------------------------------------------------------------------
# WHAM Dataset - Helper Functions
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Common - Helper Functions
# ------------------------------------------------------------------------------

def _count_len(file: str) -> float:
    """
    Count the length of a .wav file in seconds.
    
    Args:
        file (str): Path to the .wav file.

    Returns:
        float: Length of the audio file in seconds.
    """
    data, samplerate = sf.read(file)
    return len(data) / samplerate


def _reduce_dataset(df: pd.DataFrame, reduce_param: float | int | None) -> pd.DataFrame:
    """
    Reduces the dataset based on the reduce_param.
    
    Args:
        df (pd.DataFrame): The input dataframe to reduce.
        reduce_param (float | int | None): If float in (0,1], reduces the dataset to that fraction. If int > 1, reduces the dataset to that many samples. If None, no reduction is applied.

    Returns:
        pd.DataFrame: The reduced dataframe.
    """
    if reduce_param is None:
        return df
    if isinstance(reduce_param, float) and 0 < reduce_param <= 1:
        return df.sample(frac=reduce_param, random_state=42)
    if isinstance(reduce_param, int) and reduce_param > 1:
        return df.sample(n=min(reduce_param, len(df)), random_state=42)
    raise ValueError("reduce_param musi być None, float w (0,1] albo int > 1")


def _filter_dataset(df: pd.DataFrame, filter: dict[str, list[str]]) -> pd.DataFrame:
    """
    Filters the dataset based on the provided filter criteria.

    Args:
        df (pd.DataFrame): The input dataframe to filter.
        filter (dict[str, list[str]]): A dictionary where keys are column names and values are lists of acceptable values for those columns.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    for column, accepted_values in filter.items():
        df = df[df[column].isin(accepted_values)]
    return df