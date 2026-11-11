
// =================================
// AQI FORECASTER CONFIGURATION
// =================================

// Replace this with your actual deployed Streamlit URL.

const APP_URL = "https://YOUR-APP-NAME.streamlit.app";


// =================================
// DASHBOARD LINKS
// =================================

const dashboardButtons = [
    "dashboard-link",
    "hero-dashboard",
    "about-dashboard"
];

dashboardButtons.forEach((id) => {

    const button = document.getElementById(id);

    if (button) {

        button.href = APP_URL;
        button.target = "_blank";
        button.rel = "noopener noreferrer";

    }

});


// =================================
// DARK / LIGHT THEME
// =================================

const themeBtn = document.getElementById("theme-btn");

function updateThemeIcon() {

    const isDark = document.body.classList.contains("dark");

    themeBtn.innerHTML = isDark
        ? '<i data-lucide="sun"></i>'
        : '<i data-lucide="moon"></i>';

    if (window.lucide) {
        lucide.createIcons();
    }

}

themeBtn.addEventListener("click", () => {

    document.body.classList.toggle("dark");

    updateThemeIcon();

});


// =================================
// MOBILE MENU
// =================================

const menuBtn = document.getElementById("menu-btn");

const navLinks = document.querySelector(".nav-links");

menuBtn.addEventListener("click", () => {

    navLinks.classList.toggle("show");

    menuBtn.innerHTML = navLinks.classList.contains("show")
        ? '<i data-lucide="x"></i>'
        : '<i data-lucide="menu"></i>';

    if (window.lucide) {
        lucide.createIcons();
    }

});


// Close menu after clicking a link

document.querySelectorAll(".nav-links a").forEach((link) => {

    link.addEventListener("click", () => {

        navLinks.classList.remove("show");

        menuBtn.innerHTML = '<i data-lucide="menu"></i>';

        if (window.lucide) {
            lucide.createIcons();
        }

    });

});


// =================================
// FOOTER YEAR
// =================================

document.getElementById("year").textContent =
    new Date().getFullYear();


// =================================
// INITIALIZE ICONS
// =================================

if (window.lucide) {
    lucide.createIcons();
}
