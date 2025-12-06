/* ============================================
   Life, The Universe & That Safety Thing
   Main JavaScript
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
  initHeader();
  initMobileNav();
  initScrollAnimations();
});

/* ---------- Header Scroll Effect ---------- */
function initHeader() {
  const header = document.querySelector('.site-header');
  if (!header) return;
  
  let lastScroll = 0;
  const scrollThreshold = 50;
  
  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    // Add/remove scrolled class for background
    if (currentScroll > scrollThreshold) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
    
    lastScroll = currentScroll;
  }, { passive: true });
}

/* ---------- Mobile Navigation ---------- */
function initMobileNav() {
  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('.main-nav');
  
  if (!toggle || !nav) return;
  
  toggle.addEventListener('click', () => {
    nav.classList.toggle('active');
    toggle.classList.toggle('active');
    
    // Update aria-expanded
    const isExpanded = nav.classList.contains('active');
    toggle.setAttribute('aria-expanded', isExpanded);
  });
  
  // Close nav when clicking outside
  document.addEventListener('click', (e) => {
    if (!nav.contains(e.target) && !toggle.contains(e.target)) {
      nav.classList.remove('active');
      toggle.classList.remove('active');
      toggle.setAttribute('aria-expanded', 'false');
    }
  });
  
  // Close nav on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && nav.classList.contains('active')) {
      nav.classList.remove('active');
      toggle.classList.remove('active');
      toggle.setAttribute('aria-expanded', 'false');
    }
  });
}

/* ---------- Scroll Animations ---------- */
function initScrollAnimations() {
  // Elements to animate on scroll
  const animateElements = document.querySelectorAll('.card, .post-item, .project-card');
  
  if (!animateElements.length) return;
  
  // Set initial state
  animateElements.forEach((el, index) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = `opacity 0.5s ease ${index * 0.05}s, transform 0.5s ease ${index * 0.05}s`;
  });
  
  // Intersection Observer for scroll animations
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  });
  
  animateElements.forEach(el => observer.observe(el));
}

/* ---------- Utility: Set Active Nav Link ---------- */
function setActiveNavLink() {
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll('.nav-link');
  
  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (currentPath.includes(href) && href !== '/') {
      link.classList.add('active');
    } else if (currentPath === '/' && href === '/') {
      link.classList.add('active');
    }
  });
}

// Run on load
setActiveNavLink();
