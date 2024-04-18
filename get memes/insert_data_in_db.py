import mysql.connector


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="thesis"
)
mycursor = mydb.cursor()


with open(r"E:\Master\Disertatie\teste\all_photos_links.txt", "r") as file:
    links = file.readlines()


for link in links:
    link = link.strip()
    sql = "INSERT INTO pictures (link) VALUES (%s)"
    val = (link,)
    mycursor.execute(sql, val)


mydb.commit()

print(mycursor.rowcount, "Înregistrări inserate cu succes.")