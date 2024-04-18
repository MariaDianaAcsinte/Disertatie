import os


def rename_images(folder_path):
    files = os.listdir(folder_path)

    # files.sort()
    print(files)
    next_number = 1

    for filename in files:
        old_path = os.path.join(folder_path, filename)
        if os.path.isfile(old_path):
            ext = os.path.splitext(filename)[1]
            new_filename = f"{next_number:05d}{ext}"
            print(new_filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            next_number += 1

    print("Finished renaming images.")


if __name__ == "__main__":
    folder_path = r"E:\Master\Disertatie\teste\PoliticalMemes"
    rename_images(folder_path)