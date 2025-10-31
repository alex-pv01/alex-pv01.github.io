// Language management
(function() {
    'use strict';

    let currentTranslations = {};

    // Load translations from JSON file
    async function loadTranslations(lang) {
        try {
            const response = await fetch(`/translations/${lang}.json`);
            if (!response.ok) {
                console.error(`Failed to load translations for ${lang}`);
                return null;
            }
            return await response.json();
        } catch (error) {
            console.error(`Error loading translations for ${lang}:`, error);
            return null;
        }
    }

    // Get nested value from object using dot notation (e.g., "nav.home")
    function getNestedValue(obj, path) {
        return path.split('.').reduce((curr, key) => curr?.[key], obj);
    }

    // Apply translations to all elements with data-i18n attribute
    function applyTranslations(translations) {
        if (!translations) return;

        const elements = document.querySelectorAll('[data-i18n]');
        elements.forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = getNestedValue(translations, key);

            if (translation) {
                // Check if this element should use innerHTML (for content with HTML tags)
                const useHTML = element.hasAttribute('data-i18n-html');

                if (useHTML) {
                    element.innerHTML = translation;
                } else {
                    element.textContent = translation;
                }
            }
        });

        // Update document title based on current page
        updateDocumentTitle(translations);
    }

    // Update the document title based on the current page
    function updateDocumentTitle(translations) {
        const path = window.location.pathname;
        let titleKey = null;
        let descriptionKey = null;

        if (path.includes('about.html')) {
            titleKey = 'meta.titleAbout';
            descriptionKey = 'meta.aboutDescription';
        } else if (path.includes('thoughts.html')) {
            titleKey = 'meta.titleThoughts';
            descriptionKey = 'meta.thoughtsDescription';
        } else {
            // index.html
            descriptionKey = 'meta.description';
        }

        // Update title
        if (titleKey) {
            const title = getNestedValue(translations, titleKey);
            if (title) {
                document.title = title;
            }
        }

        // Update meta description
        if (descriptionKey) {
            const description = getNestedValue(translations, descriptionKey);
            if (description) {
                const metaDescription = document.querySelector('meta[name="description"]');
                if (metaDescription) {
                    metaDescription.setAttribute('content', description);
                }
            }
        }
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

    // Switch language
    async function switchLanguage(lang) {
        localStorage.setItem('language', lang);
        updateLanguageUI(lang);

        const translations = await loadTranslations(lang);
        if (translations) {
            currentTranslations = translations;
            applyTranslations(translations);
            console.log(`Language switched to: ${lang}`);
        }
    }

    // Initialize language
    async function initLanguage() {
        const savedLang = localStorage.getItem('language') || 'en';
        updateLanguageUI(savedLang);

        const translations = await loadTranslations(savedLang);
        if (translations) {
            currentTranslations = translations;
            applyTranslations(translations);
        }
    }

    // Set up language selector when DOM is ready
    function setupLanguageSelector() {
        const langSpans = document.querySelectorAll('.language-selector span[data-lang]');

        langSpans.forEach(span => {
            span.addEventListener('click', function() {
                const lang = this.getAttribute('data-lang');
                switchLanguage(lang);
            });
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            initLanguage();
            setupLanguageSelector();
        });
    } else {
        initLanguage();
        setupLanguageSelector();
    }
})();
