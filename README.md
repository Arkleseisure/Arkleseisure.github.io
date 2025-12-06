# Life, The Universe & That Safety Thing

A personal website for essays on philosophy, AI safety, and everything in between.

## Structure

```
├── index.html          # Homepage
├── life.html           # Life section (course-style articles)
├── universe.html       # Universe section (misc essays)
├── safety.html         # AI Safety section (articles)
├── projects.html       # AI Safety projects
├── about.html          # About page
├── css/
│   └── style.css       # All styles
├── js/
│   └── main.js         # Navigation & interactions
├── posts/
│   ├── life/           # Life articles
│   │   └── _template.html
│   ├── universe/       # Universe articles
│   └── safety/         # AI Safety articles
└── images/             # Any images you add
```

## Adding a New Article

### 1. Create the article file

Copy the template from `posts/life/_template.html` to the appropriate folder:

```bash
cp posts/life/_template.html posts/life/my-new-article.html
```

### 2. Edit the article

Open the file and update:
- `<title>` tag
- `<meta name="description">` 
- The section label (e.g., "Life · Article 01")
- The `<h1>` title
- The date and reading time
- All the content in `<div class="article-content">`
- The previous/next navigation links

### 3. Add to the section page

Open the relevant section page (e.g., `life.html`) and add a link in the post list:

```html
<a href="posts/life/my-new-article.html" class="post-item">
  <span class="post-number">01</span>
  <span class="post-title">My New Article Title</span>
  <span class="post-date">Dec 2024</span>
</a>
```

For the Life section specifically, use sequential numbers (01, 02, 03...) to indicate reading order.

For Universe and AI Safety, you can use "—" or omit the number.

### 4. (Optional) Add to recent posts on homepage

Open `index.html` and add to the "Recent Writing" section.

## Customization

### Colors

Edit the CSS variables at the top of `css/style.css`:

```css
:root {
  --accent-life: #4ecdc4;      /* Teal */
  --accent-universe: #a855f7;   /* Purple */
  --accent-safety: #f59e0b;     /* Amber */
  /* ... */
}
```

### Fonts

The site uses Google Fonts. To change them, edit the `<link>` tag in each HTML file's `<head>` and update the CSS variables:

```css
:root {
  --font-display: 'Playfair Display', Georgia, serif;
  --font-body: 'Source Sans 3', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}
```

### Adding fancy effects later

The site is pure HTML/CSS/JS with no framework, so you can add anything:

- **Parallax scrolling**: Add scroll event listeners in `main.js`
- **Animated backgrounds**: Add canvas elements or CSS animations
- **Page transitions**: Use the View Transitions API or a library like Barba.js
- **3D effects**: Add Three.js or similar

## Deployment

This is designed for GitHub Pages. Just push to your repository and it will deploy automatically.

The `CNAME` file is set to `lifeuniversesafety.com` — update this if your domain is different.

## Fill in placeholders

Search for `[` in the HTML files to find all placeholder content that needs to be filled in:

- Homepage intro about you
- About page bio
- Contact links
- Any placeholder articles

---

Built with care. Now go write something worth reading.
