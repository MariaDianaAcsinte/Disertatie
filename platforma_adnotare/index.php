<?php
session_start(); // Pornirea sesiunii

// Verificarea dacă utilizatorul este autentificat
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    // Dacă utilizatorul nu este autentificat, redirecționați către pagina de login/register
    header("Location: login_register.php");
    exit;
}

// Conexiunea la baza de date
$servername = "sql308.infinityfree.com";
$username = "if0_36252006";
$password = "DVrqNTYvOw1xXWO";
$database = "if0_36252006_thesis";

$conn = new mysqli($servername, $username, $password, $database);
if ($conn->connect_error) {
    die("Conexiunea la baza de date a eșuat: " . $conn->connect_error);
}

$user_id = $_SESSION['user_id'];

// Interogarea SQL pentru a selecta o imagine pentru care utilizatorul nu a trimis deja un răspuns
$sql_images = "SELECT p.id, p.link 
               FROM pictures p 
               LEFT JOIN answers a ON p.id = a.picture_id AND a.user_id = $user_id
               WHERE a.id IS NULL";

// Numărul total de poze rămase pentru utilizator
$result_count = $conn->query("SELECT COUNT(*) as total_images FROM ($sql_images) AS temp");
$row_count = $result_count->fetch_assoc();
$total_images = $row_count["total_images"];

// Obținerea unei poze random pentru utilizator
$sql_random_image = "$sql_images ORDER BY RAND() LIMIT 1";
$result = $conn->query($sql_random_image);
if ($result->num_rows > 0) {
    // Afisarea pozei random
    $row = $result->fetch_assoc();
    $random_image_id = $row["id"];
    $random_image_link = $row["link"];
    ?>
    <style>
        .container {
            display: flex;
            align-items: flex-start; /* Aliniere sus */
        }
        .image-container {
            margin-right: 20px; /* Spațiu între imagine și text */
        }
        .question {
            align-self: flex-start; /* Aliniere sus */
            margin-top: 0; /* Eliminare margin sus */
            font-size: 24px; /* Dimensiunea textului întrebării */
            font-weight: bold; /* Stil bold pentru întrebare */
            color: #333; /* Culoarea textului întrebării */
        }
        .answer {
            font-size: 18px; /* Dimensiunea textului răspunsului */
            color: #666; /* Culoarea textului răspunsului */
        }
    </style>
    <div class="container">
        <div class="image-container">
            <img src="<?php echo $random_image_link; ?>" alt="Random Image"><br><br>
        </div>
        <form action="save_answer.php" method="post">
            <input type="hidden" name="image_id" value="<?php echo $random_image_id; ?>">
            <label for="answer" class="question">Is ironic?</label><br>
            <input type="radio" id="yes" name="answer" value="1" required>
            <label for="yes" class="answer">Yes</label><br>
            <input type="radio" id="no" name="answer" value="0" required>
            <label for="no" class="answer">No</label><br><br>
            <button type="submit">Submit</button>
        </form>
    </div>
    <p><?php echo "You have $total_images memes to run!"; ?></p>
    <?php
} else {
    echo "No more memes for you!";
}

// Închiderea conexiunii cu baza de date
$conn->close();
?>
