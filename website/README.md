# OpenFinOps Website

Professional marketing website for OpenFinOps - AI/ML Cost Intelligence Platform.

## 📁 Structure

```
website/
├── index.html              # Landing page
├── features.html           # Features overview
├── documentation.html      # Getting started & usage docs
├── api.html               # API reference
├── demo.html              # Live demo page
├── css/
│   └── style.css          # Main stylesheet
├── js/
│   └── main.js            # JavaScript utilities
└── images/                # Image assets
```

## 🎨 Design

- **Theme**: Dark with glassmorphism effects
- **Colors**:
  - Primary: #00d4aa (Cyan)
  - Secondary: #00a8ff (Blue)
  - Accent: #3742fa (Purple)
- **Typography**: Inter font family
- **Framework**: Vanilla HTML/CSS/JS with Font Awesome icons

## 📄 Pages

### 1. Home (index.html)
- Hero section with key value propositions
- Feature preview cards
- Use cases for different roles
- Technology stack showcase
- Call-to-action sections

### 2. Features (features.html)
- Detailed feature explanations
- AI/ML cost tracking
- Multi-cloud support
- Executive dashboards
- Smart recommendations
- Alerting & monitoring
- Telemetry agents

### 3. Documentation (documentation.html)
- Getting started guide
- Installation instructions
- Quick start tutorial
- Web UI server setup
- Observability Hub usage
- Dashboard examples
- Telemetry agent deployment
- Configuration options

### 4. API Reference (api.html)
- Complete API documentation
- ObservabilityHub methods
- LLM Observability APIs
- Cost Observatory APIs
- Dashboard APIs
- Telemetry agent APIs
- Web UI server APIs
- Code examples for each method

### 5. Live Demo (demo.html)
- Links to live Web UI dashboards
- Overview dashboard
- CFO executive dashboard
- COO operational dashboard
- Infrastructure dashboard
- Quick start guide
- System requirements

## 🚀 Usage

### Development
1. Open `index.html` directly in a browser
2. Or use a local server:
   ```bash
   # Python 3
   python -m http.server 8000

   # Python 2
   python -m SimpleHTTPServer 8000

   # Node.js
   npx http-server -p 8000
   ```
3. Navigate to `http://localhost:8000`

### Deployment

#### GitHub Pages
1. Push to GitHub repository
2. Go to Settings > Pages
3. Set source to `main` branch `/website` folder
4. Site will be available at `https://username.github.io/openfinops/`

#### Netlify
1. Connect your GitHub repository
2. Set build directory to `website`
3. Deploy

#### Custom Server
1. Copy all files to web server directory
2. Configure web server to serve static files
3. Ensure proper MIME types for CSS/JS

## 🔗 Links

All demo links point to `http://localhost:8080` by default. Update these in `demo.html` for production:

```html
<!-- Update these URLs for production -->
<a href="http://localhost:8080/" ...>
<a href="http://localhost:8080/dashboard/cfo" ...>
<a href="http://localhost:8080/dashboard/coo" ...>
<a href="http://localhost:8080/dashboard/infrastructure" ...>
```

## 🎯 Features

- ✅ Fully responsive design
- ✅ Modern glassmorphism UI
- ✅ Smooth animations
- ✅ Mobile-friendly navigation
- ✅ Fast loading (no heavy frameworks)
- ✅ SEO optimized
- ✅ Accessible design
- ✅ Cross-browser compatible

## 📝 Customization

### Colors
Edit CSS variables in `css/style.css`:
```css
:root {
    --primary-color: #00d4aa;
    --secondary-color: #00a8ff;
    --accent-color: #3742fa;
    /* ... */
}
```

### Content
- Update text in respective HTML files
- Modify feature cards in `features.html`
- Update API examples in `api.html`
- Change documentation in `documentation.html`

### Styling
- Main styles: `css/style.css`
- Page-specific styles: `<style>` tags in each HTML file
- JavaScript: `js/main.js`

## 🔧 Dependencies

External CDN resources:
- Font Awesome 6.4.0 (icons)
- Google Fonts (Inter font family)
- Fira Code (monospace for code)

All dependencies loaded via CDN, no build process required.

## 📱 Responsive Breakpoints

- Desktop: > 768px
- Tablet: 768px
- Mobile: < 768px

## 🌐 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Opera (latest)

## 📄 License

Apache-2.0 License - See LICENSE file in project root.

## 🤝 Contributing

To contribute to the website:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on multiple browsers
5. Submit a pull request

## 📧 Contact

For issues or questions:
- GitHub: https://github.com/rdmurugan/openfinops
- Issues: https://github.com/rdmurugan/openfinops/issues
