import datetime
import os
import base64
import time

import requests
from PIL import Image
import traceback

path = r'E:\Master\Disertatie\teste\PoliticalMemes'
api_link = 'https://api.ocr.space/parse/image'
apiKey = '1e2e44e4ad88957'
text_folder = r'E:\Master\Disertatie\teste\text_for_memes'

for file in os.listdir(path):
    print(file)
    meme_path = os.path.join(path, file)
    file_name_ = file.split('.')[0] + '.txt'
    file_name_ = os.path.join(text_folder, file_name_)
    if os.path.exists(file_name_):
        continue

    # Check if the file exists and is not empty
    if os.path.isfile(meme_path) and os.path.getsize(meme_path) > 0:
        with open(meme_path, 'rb') as image_file:
            # Convert image to base64 string
            base64_bytes = base64.b64encode(image_file.read())
            # base64_string = base64_bytes.decode('utf-8')
            base64_string = 'data:image/jpeg;base64,' + base64_bytes.decode('utf-8')
            # img = Image.open(meme_path).

            # Make API request
            post_data = {
                "apikey": apiKey,
                "language": "eng",
                "detectOrientation": True,
                "OCREngine": 2,
                "scale": True,
                "base64Image": base64_string,
                "filetype": "jpg"  # Adjust based on your file format
            }
            # files = {'file': (file, img, 'image/jpeg')}
            response = requests.post(api_link, data=post_data)
            try:
                jsonData = response.json()
                if ' 180 number of times within 3600 seconds' in jsonData:
                    print("Ban o ora: {}".format(datetime.datetime.now()))
                    time.sleep(3630)
                print("Aici", jsonData)
                text = jsonData['ParsedResults'][0]['ParsedText']
                file_name = file.split('.')[0] + '.txt'
                file_name = os.path.join(text_folder, file_name)
                with open(file_name, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                print(text)
            except Exception as e:
                print(e, meme_path)
                traceback.print_exc()
                pass

    else:
        print(f"File '{file}' is empty or does not exist.")
    # exit()
