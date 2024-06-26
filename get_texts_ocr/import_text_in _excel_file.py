import pandas as pd
import os
from openpyxl import Workbook

from openpyxl.utils.dataframe import dataframe_to_rows

# Directorul în care sunt stocate documentele text
dir_path = r'E:\Master\Disertatie\teste\text_for_memes'

# Lista pentru a stoca conținutul documentelor text
document_contents = []

# Parcurgem toate documentele text și le citim conținutul
for i in range(1, 3588):
    i = str(i).zfill(5)
    file_name = f'{i}.txt'
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            document_contents.append({'ID': i, 'Content': content})
    else:
        print(f"Documentul {file_name} lipsește.")

# Convertim lista de dicționare într-un DataFrame pandas
df = pd.DataFrame(document_contents)

wb = Workbook()
ws = wb.active

for r in dataframe_to_rows(df, index=False, header=True):
    ws.append(r)
# Salvăm DataFrame-ul într-un fișier Excel
# output_file = 'Texts_from_memes.xlsx'
# df.to_excel(output_file, index=False)

wb.save("Texts_from_memes.xlsx")

print("Documentele au fost importate cu succes în fișierul ala.")
