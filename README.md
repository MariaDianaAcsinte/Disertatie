1. Data set preparation

First I preprocessed the dataset, i.e. I made a script that downloads from Reddit from several subreddits (PoliticalMemes, Political Humor) and save them 
in my disertation folder.
Then I created another script that removes duplicates, keeping only an unic file.
I then created a script to rename the files in order
because the jpeg or png dataset would have been too large to upload to a web host 
I converted all jpg/jpeg or png files to urls and saved the list, then inserted them into a MySQL database, 
simplifying data management and preparing them for later use in the annotation platform.

2. Annotation platform

In order to display memes, the code connects to a MySQL database, starts a PHP session for user authentication, and determines whether the user is logged in.
Random memes are chosen, users are questioned about them, the answers are saved, new memes are displayed, and the connection is closed. 
This straightforward meme grading software improves user experience and adds value to the platform.

It includes backend functionality to manage these operations as well as an HTML page for registration and authentication. 
The HTML page has links for both registration and login, along with two forms for registration and authentication. 
When a user clicks the "Login" or "Register" buttons, the JavaScript script hides the form, enabling navigation without requiring a page refresh. 
After processing the form data, the PHP script verifies the action entered in the hidden "action" field and performs the appropriate action. 
Details are used to establish the interaction with the MySQL database.

Then adding an additional script that starts a session, verifies user identity, connects to a MySQL database, and ends the process if a connection
cannot be made. After processing data received via POST form, the script extracts user ID and form data. To add a response to the database, a SQL query is run,
followed by a redirect to index.php and confirmation. The database connection has been terminated.

3. OCR Task

This Python script extracts text from images using the OCR service provided by the OCR.Space API. It sets a working directory and a text_folder for the
extracted text files. The script checks if a text file exists for each image and converts it to a base64 encoded string. The text is sent to the OCR.
Space service using an HTTP POST request, and the script waits for a specified time if the service usage limit is reached. 
The process is repeated for each image in the path directory.

4. ML Task

With the text extracted from the memes I tried running the code Multimodal Meme Dataset (MultiOFF) for Identifying Offensive Content in Image and Text and got the following results: 

Epoch 1/7
2/2 [==============================] - 12s 9s/step - loss: 0.7925 - accuracy: 0.6406 - val_loss: 0.6887 - val_accuracy: 0.5839
Epoch 2/7
2/2 [==============================] - 10s 9s/step - loss: 0.8029 - accuracy: 0.6094 - val_loss: 0.6674 - val_accuracy: 0.6980
Epoch 3/7
2/2 [==============================] - 9s 9s/step - loss: 0.8388 - accuracy: 0.5156 - val_loss: 0.6889 - val_accuracy: 0.5839
Epoch 4/7
2/2 [==============================] - 7s 7s/step - loss: 0.8002 - accuracy: 0.6094 - val_loss: 0.6911 - val_accuracy: 0.6980
Epoch 5/7
2/2 [==============================] - 5s 5s/step - loss: 0.8015 - accuracy: 0.6094 - val_loss: 0.6929 - val_accuracy: 0.5839
Epoch 6/7
2/2 [==============================] - 5s 5s/step - loss: 0.7744 - accuracy: 0.7031 - val_loss: 0.6882 - val_accuracy: 0.6980
Epoch 7/7
2/2 [==============================] - 5s 5s/step - loss: 0.8004 - accuracy: 0.6094 - val_loss: 0.6878 - val_accuracy: 0.5839
