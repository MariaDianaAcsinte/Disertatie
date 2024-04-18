<?php
session_start(); // Pornirea sesiunii

error_reporting(E_ALL);
ini_set('display_errors', 1);

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

// Preiați datele din formular


if ($_SERVER["REQUEST_METHOD"] == "POST") {
    echo "Valorile primite prin formular sunt:<br>";
        print_r($_POST);

    // Preiați alte date din formular (id-ul imaginii și răspunsul)
    $picture_id = $_POST['image_id'];
    $response = $_POST['answer'];
    $user_id = $_SESSION['user_id']; 

    echo "Id-ul imaginii: $picture_id<br>";
    echo "Răspunsul: $response<br>";
    echo "Id-ul utilizatorului: $user_id<br>";

    // Adăugați răspunsul în baza de date
    $insert_query = "INSERT INTO answers (user_id, picture_id, response) VALUES ('$user_id', '$picture_id', '$response')";
    if ($conn->query($insert_query) === TRUE) {
        echo "Răspunsul a fost salvat cu succes în baza de date.";
        header("Location: index.php");
    } else {
        echo "Eroare la salvarea răspunsului în baza de date: " . $conn->error;
        header("Location: index.php");
    }
}

// Închiderea conexiunii cu baza de date
$conn->close();
?>
