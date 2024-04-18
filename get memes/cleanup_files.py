import os
import hashlib
from collections import defaultdict


def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def remove_duplicates(folder_path):
    hash_map = defaultdict(list)

    for filename in os.listdir(folder_path):
        print(filename)
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_hash = get_file_hash(file_path)
            hash_map[file_hash].append(filename)

    for hash_value, filenames in hash_map.items():
        if len(filenames) > 1:
            print(f"Duplicate found: {filenames}")
            for i, filename in enumerate(filenames[1:], start=2):
                os.remove(os.path.join(folder_path, filename))


if __name__ == "__main__":
    folder_path = r"E:\Master\Disertatie\teste\PoliticalMemes"
    remove_duplicates(folder_path)