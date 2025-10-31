// Mobile menu toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    const menuToggle = document.getElementById('mobile-menu-toggle');

    if (menuToggle) {
        menuToggle.addEventListener('click', function() {
            document.body.classList.toggle('mobile-menu-open');
        });

        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            const isMenuButton = event.target.closest('#mobile-menu-toggle');
            const isThemeToggle = event.target.closest('#theme-toggle');
            const isLanguageSelector = event.target.closest('.language-selector');

            // If click is outside menu button and menu items, close the menu
            if (!isMenuButton && !isThemeToggle && !isLanguageSelector) {
                document.body.classList.remove('mobile-menu-open');
            }
        });
    }
});
