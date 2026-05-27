import os


def create_filepaths(root_dir: str, subset: str, extension: str = ".wav") -> dict[str, list[str]]:
    """Create lists of file paths for noisy and clean audio files in the specified subset."""
    subset_path = os.path.join(root_dir, subset)
    noisy_dir = os.path.join(subset_path, "noisy")
    clean_dir = os.path.join(subset_path, "clean")
    noisy_persons = [os.path.join(noisy_dir, d) for d in os.listdir(noisy_dir) if os.path.isdir(os.path.join(noisy_dir, d))]
    clean_persons = [os.path.join(clean_dir, d) for d in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir, d))]
    noisy_files = []
    clean_files = []
    for person_dir in noisy_persons:
        person_noisy_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(extension)]
        noisy_files.extend(person_noisy_files)
    
    for person_dir in clean_persons:
        person_clean_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(extension)]
        clean_files.extend(person_clean_files)
    
    return {
        "noisy": sorted(noisy_files),
        "clean": sorted(clean_files)
    }