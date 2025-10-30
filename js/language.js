// Language management
(function() {
    'use strict';

    // Initialize language immediately
    function initLanguage() {
        const savedLang = localStorage.getItem('language') || 'en';
        updateLanguageUI(savedLang);
        return savedLang;
    }

    // Update UI to reflect current language
    function updateLanguageUI(lang) {
        const langSpans = document.querySelectorAll('.language-selector span[data-lang]');
        langSpans.forEach(span => {
            const spanLang = span.getAttribute('data-lang');
            // Remove all active classes
            span.classList.remove('active-en', 'active-cat', 'active-dk');

            // Add appropriate active class if this is the selected language
            if (spanLang === lang) {
                span.classList.add(`active-${lang}`);
            }
        });
    }

    // Set up language selector when DOM is ready
    function setupLanguageSelector() {
        const langSpans = document.querySelectorAll('.language-selector span[data-lang]');

        langSpans.forEach(span => {
            span.addEventListener('click', function() {
                const lang = this.getAttribute('data-lang');
                localStorage.setItem('language', lang);
                updateLanguageUI(lang);

                // Here you can add logic to switch content based on language
                // For now, it just stores the preference
                console.log(`Language switched to: ${lang}`);
            });
        });
    }

    // Initialize immediately
    initLanguage();

    // Setup selector when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupLanguageSelector);
    } else {
        setupLanguageSelector();
    }
})();
