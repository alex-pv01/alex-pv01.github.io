// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add current year to any element with id "current-year" if needed
    const currentYearElements = document.querySelectorAll('.current-year');
    if (currentYearElements.length > 0) {
        const currentYear = new Date().getFullYear();
        currentYearElements.forEach(element => {
            element.textContent = currentYear;
        });
    }

    // Highlight the current section in the navigation
    function highlightNav() {
        const sections = document.querySelectorAll('section');
        const navLinks = document.querySelectorAll('.nav-link');
        
        // Get the current scroll position
        const scrollPosition = window.scrollY;
        
        // Find the section that's currently in view
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.clientHeight;
            const sectionId = section.getAttribute('id');
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                // Remove 'active' class from all links
                navLinks.forEach(link => {
                    link.classList.remove('active');
                });
                
                // Add 'active' class to the corresponding navigation link
                const currentNavLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);
                if (currentNavLink) {
                    currentNavLink.classList.add('active');
                }
            }
        });
    }

    // Listen for scroll events
    window.addEventListener('scroll', highlightNav);
    
    // Initial call to highlight the correct navigation item
    highlightNav();
});