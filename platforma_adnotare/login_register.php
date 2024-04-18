<!-- login_register.php -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login/Register</title>
    <style>
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <h2>Login/Register</h2>

    <!-- Butonul pentru login -->
    <button id="loginBtn">Login</button>
    <!-- Butonul pentru înregistrare -->
    <button id="registerBtn">Register</button>

    <!-- Formularul pentru autentificare -->
    <form id="loginForm" action="login_register.php" method="post" class="hidden">
        <input type="hidden" name="action" value="login"> <!-- Adăugați un câmp ascuns pentru a identifica acțiunea -->
        <label for="login_username">Username:</label>
        <input type="text" id="login_username" name="login_username" required><br><br>
        <label for="login_password">Password:</label>
        <input type="password" id="login_password" name="login_password" required><br><br>
        <button type="submit">Login</button>
    </form>

    <!-- Formularul pentru înregistrare -->
    <form id="registerForm" action="login_register.php" method="post" class="hidden">
        <input type="hidden" name="action" value="register"> <!-- Adăugați un câmp ascuns pentru a identifica acțiunea -->
        <label for="register_username">Username:</label>
        <input type="text" id="register_username" name="register_username" required><br><br>
        <label for="register_password">Password:</label>
        <input type="password" id="register_password" name="register_password" required><br><br>
        <label for="confirm_password">Confirm Password:</label>
        <input type="password" id="confirm_password" name="confirm_password" required><br><br>
        <button type="submit">Register</button>
    </form>

    <script>
        // Obțineți referințe către butoane și formulare
        const loginBtn = document.getElementById("loginBtn");
        const registerBtn = document.getElementById("registerBtn");
        const loginForm = document.getElementById("loginForm");
        const registerForm = document.getElementById("registerForm");

        // Ascundeți formularul de înregistrare inițial
        registerForm.classList.add("hidden");

        // Ascultați evenimentele de clic pentru butoanele de login și înregistrare
        loginBtn.addEventListener("click", function() {
            // Ascundeți formularul de înregistrare și afișați formularul de login
            registerForm.classList.add("hidden");
            loginForm.classList.remove("hidden");
        });

        registerBtn.addEventListener("click", function() {
            // Ascundeți formularul de login și afișați formularul de înregistrare
            loginForm.classList.add("hidden");
            registerForm.classList.remove("hidden");
        });
    </script>

    <?php
// Conexiunea la baza de date
session_start();
$servername = "sql308.infinityfree.com";
$username = "if0_36252006";
$password = "DVrqNTYvOw1xXWO";
$database = "if0_36252006_thesis";

$conn = new mysqli($servername, $username, $password, $database);
if ($conn->connect_error) {
    die("Conexiunea la baza de date a eșuat: " . $conn->connect_error);
}

// Procesarea datelor formularului
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $action = $_POST["action"];

    // Verificăm acțiunea și tratăm-o în consecință
    if ($action == "login") {
        // Procesăm autentificarea
        $login_username = $_POST["login_username"];
        $login_password = $_POST["login_password"];

        // Verificăm dacă utilizatorul există în baza de date
        $sql = "SELECT id FROM users WHERE username='$login_username' AND password='$login_password'";
        $result = $conn->query($sql);

        if ($result->num_rows > 0) {
            // Utilizatorul a fost autentificat cu succes
            $row = $result->fetch_assoc();
            $user_id = $row['id']; // Salvăm id-ul utilizatorului

            $_SESSION['loggedin'] = true;
            $_SESSION['user_id'] = $user_id;
            echo "Autentificare reușită!";
            // Redirecționăm către pagina principală sau unde doriți
            header("Location: index.php");
            exit();
        } else {
            // Autentificare eșuată
            echo "Nume de utilizator sau parolă incorecte!";
        }
    } elseif ($action == "register") {
        // Procesăm înregistrarea
        $register_username = $_POST["register_username"];
        $register_password = $_POST["register_password"];
        $confirm_password = $_POST["confirm_password"];

        // Verificăm dacă parola și confirmarea parolei coincid
        if ($register_password !== $confirm_password) {
            echo "Parola și confirmarea parolei nu corespund!";
        } else {
            // Verificăm dacă utilizatorul există deja în baza de date
            $check_user_sql = "SELECT * FROM users WHERE username='$register_username'";
            $check_user_result = $conn->query($check_user_sql);

            if ($check_user_result->num_rows > 0) {
                echo "Numele de utilizator este deja folosit!";
            } else {
                // Adăugăm noul utilizator în baza de date
                $register_sql = "INSERT INTO users (username, password) VALUES ('$register_username', '$register_password')";
                if ($conn->query($register_sql) === TRUE) {
                    $user_id = $conn->insert_id;
                    $_SESSION['user_id'] = $user_id;
                    $_SESSION['loggedin'] = true;
                    echo "Înregistrare reușită!";
                    header("Location: index.php");
                    exit();
                } else {
                    echo "Eroare la înregistrare: " . $conn->error;
                }
            }
        }
    }
}
?>

</body>
</html>
