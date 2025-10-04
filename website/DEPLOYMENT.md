# OpenFinOps Website Deployment Guide

Complete guide for deploying the OpenFinOps marketing website with live demo integration.

## 📋 Prerequisites

- OpenFinOps server running (for live demo)
- Git repository (for GitHub Pages/Netlify/Vercel)
- Domain name (optional, for custom domains)

## 🚀 Deployment Options

### Option 1: GitHub Pages (Recommended for Open Source)

**Pros:** Free hosting, automatic SSL, easy setup
**Cons:** Static only (demo links to external server)

#### Step 1: Prepare Repository

```bash
# Ensure website files are in /website directory
cd /Users/durai/Documents/GitHub/OpenFinOps

# Commit website files
git add website/
git commit -m "Add OpenFinOps marketing website"
git push origin main
```

#### Step 2: Configure GitHub Pages

1. Go to your repository on GitHub: `https://github.com/rdmurugan/openfinops`
2. Click **Settings** → **Pages**
3. Under **Source**, select:
   - Branch: `main`
   - Folder: `/website`
4. Click **Save**

#### Step 3: Update Demo URLs

Edit `website/demo.html` to point to your live OpenFinOps server:

```html
<!-- Update these URLs -->
<a href="https://demo.openfinops.io/" class="demo-card-link">
<a href="https://demo.openfinops.io/dashboard/cfo" class="demo-card-link">
<a href="https://demo.openfinops.io/dashboard/coo" class="demo-card-link">
<a href="https://demo.openfinops.io/dashboard/infrastructure" class="demo-card-link">
```

#### Step 4: Configure Custom Domain (Optional)

1. Add a `CNAME` file in `/website`:
   ```bash
   echo "www.openfinops.io" > website/CNAME
   ```

2. Configure DNS:
   ```
   Type: CNAME
   Name: www
   Value: rdmurugan.github.io
   ```

3. In GitHub Pages settings, add custom domain: `www.openfinops.io`

**Your site will be live at:** `https://rdmurugan.github.io/openfinops/`

---

### Option 2: Netlify (Recommended for Custom Domain)

**Pros:** Free tier, automatic deployments, serverless functions
**Cons:** None for static sites

#### Step 1: Create netlify.toml

```bash
cat > netlify.toml << 'EOF'
[build]
  publish = "website"
  command = "echo 'No build required'"

[[redirects]]
  from = "/demo/*"
  to = "https://demo.openfinops.io/:splat"
  status = 200
  force = true

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
EOF
```

#### Step 2: Deploy to Netlify

**Option A: Via Netlify UI**

1. Go to https://app.netlify.com/
2. Click "Add new site" → "Import an existing project"
3. Connect to GitHub: `rdmurugan/openfinops`
4. Configure:
   - Base directory: `website`
   - Build command: (leave empty)
   - Publish directory: `.`
5. Click "Deploy site"

**Option B: Via Netlify CLI**

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login to Netlify
netlify login

# Deploy from website directory
cd website
netlify deploy --prod
```

#### Step 3: Custom Domain

1. In Netlify dashboard → Domain settings
2. Add custom domain: `openfinops.io`
3. Follow DNS configuration instructions

**Your site will be live at:** `https://openfinops.io/` or `https://[site-name].netlify.app/`

---

### Option 3: Vercel (Great for Performance)

**Pros:** Edge network, instant deployments, free SSL
**Cons:** None

#### Step 1: Create vercel.json

```bash
cat > vercel.json << 'EOF'
{
  "version": 2,
  "public": true,
  "cleanUrls": true,
  "trailingSlash": false,
  "rewrites": [
    {
      "source": "/demo/:path*",
      "destination": "https://demo.openfinops.io/:path*"
    }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "1; mode=block"
        }
      ]
    }
  ]
}
EOF
```

#### Step 2: Deploy to Vercel

**Option A: Via Vercel UI**

1. Go to https://vercel.com/
2. Import Git repository: `rdmurugan/openfinops`
3. Configure:
   - Root Directory: `website`
   - Framework Preset: Other
4. Deploy

**Option B: Via Vercel CLI**

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy from website directory
cd website
vercel --prod
```

**Your site will be live at:** `https://openfinops.vercel.app/`

---

### Option 4: Docker Deployment (Self-Hosted)

**Pros:** Full control, can bundle with OpenFinOps server
**Cons:** Requires server management

#### Step 1: Create Dockerfile

```bash
cat > website/Dockerfile << 'EOF'
FROM nginx:alpine

# Copy website files
COPY . /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
EOF
```

#### Step 2: Create nginx.conf

```bash
cat > website/nginx.conf << 'EOF'
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/css application/javascript application/json image/svg+xml;
    gzip_comp_level 6;

    # Security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Cache static assets
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # HTML files - no cache
    location ~* \.html$ {
        expires -1;
        add_header Cache-Control "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0";
    }

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy demo requests to OpenFinOps server
    location /demo/ {
        proxy_pass http://openfinops-server:8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF
```

#### Step 3: Build and Run Docker Container

```bash
# Build image
cd website
docker build -t openfinops-website .

# Run container
docker run -d \
  --name openfinops-website \
  -p 80:80 \
  --link openfinops-server:openfinops-server \
  openfinops-website

# Or use Docker Compose (see below)
```

#### Step 4: Docker Compose (Full Stack)

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # OpenFinOps Server
  openfinops-server:
    build: .
    container_name: openfinops-server
    ports:
      - "8080:8080"
    environment:
      - OPENFINOPS_HOST=0.0.0.0
      - OPENFINOPS_PORT=8080
      - OPENFINOPS_DEBUG=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Website
  website:
    build: ./website
    container_name: openfinops-website
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - openfinops-server
    volumes:
      - ./certs:/etc/nginx/certs:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  certs:
EOF

# Start full stack
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Your site will be live at:** `http://your-server-ip/`

---

### Option 5: AWS S3 + CloudFront

**Pros:** Scalable, fast CDN, cheap
**Cons:** AWS configuration required

#### Step 1: Create S3 Bucket

```bash
# Create bucket
aws s3 mb s3://openfinops-website --region us-east-1

# Configure for static website hosting
aws s3 website s3://openfinops-website \
  --index-document index.html \
  --error-document index.html

# Set bucket policy
cat > bucket-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::openfinops-website/*"
    }
  ]
}
EOF

aws s3api put-bucket-policy \
  --bucket openfinops-website \
  --policy file://bucket-policy.json
```

#### Step 2: Upload Website

```bash
# Sync website files
cd website
aws s3 sync . s3://openfinops-website \
  --exclude ".git/*" \
  --exclude "*.md" \
  --cache-control "public, max-age=31536000" \
  --exclude "*.html"

# HTML files without cache
aws s3 sync . s3://openfinops-website \
  --exclude "*" \
  --include "*.html" \
  --cache-control "no-cache, no-store, must-revalidate"
```

#### Step 3: Create CloudFront Distribution

```bash
# Create CloudFront distribution
aws cloudfront create-distribution \
  --origin-domain-name openfinops-website.s3-website-us-east-1.amazonaws.com \
  --default-root-object index.html

# Note the distribution domain name (e.g., d1234567890.cloudfront.net)
```

#### Step 4: Configure SSL Certificate

```bash
# Request SSL certificate (in us-east-1 for CloudFront)
aws acm request-certificate \
  --domain-name openfinops.io \
  --subject-alternative-names www.openfinops.io \
  --validation-method DNS \
  --region us-east-1

# Follow DNS validation instructions
# Then update CloudFront distribution with certificate ARN
```

**Your site will be live at:** `https://[cloudfront-id].cloudfront.net/`

---

## 🔗 Connecting Live Demo

### Option 1: Separate Server (Recommended)

Run OpenFinOps server on a dedicated instance:

```bash
# On demo server (e.g., demo.openfinops.io)
docker run -d \
  --name openfinops-server \
  -p 8080:8080 \
  -e OPENFINOPS_HOST=0.0.0.0 \
  -e OPENFINOPS_PORT=8080 \
  openfinops/openfinops:latest

# Configure reverse proxy with SSL
```

**Nginx reverse proxy config:**

```nginx
server {
    listen 80;
    server_name demo.openfinops.io;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name demo.openfinops.io;

    ssl_certificate /etc/letsencrypt/live/demo.openfinops.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/demo.openfinops.io/privkey.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for live updates
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

### Option 2: Serverless Demo (AWS Lambda)

Deploy OpenFinOps as serverless function:

```bash
# Install Serverless Framework
npm install -g serverless

# Create serverless.yml
cat > serverless.yml << 'EOF'
service: openfinops-demo

provider:
  name: aws
  runtime: python3.11
  region: us-east-1

functions:
  api:
    handler: wsgi_handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true

plugins:
  - serverless-wsgi
  - serverless-python-requirements

custom:
  wsgi:
    app: src.openfinops.webui.server.app
EOF

# Deploy
serverless deploy
```

### Option 3: Local Development Demo

For testing, run locally:

```bash
# Terminal 1: Start OpenFinOps server
cd /Users/durai/Documents/GitHub/OpenFinOps
python -m openfinops.webui.server --host 0.0.0.0 --port 8080

# Terminal 2: Start website server
cd website
python -m http.server 8888

# Access:
# Website: http://localhost:8888
# Demo: http://localhost:8080
```

---

## 🔒 SSL Certificate Setup

### Using Let's Encrypt (Certbot)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d openfinops.io -d www.openfinops.io -d demo.openfinops.io

# Auto-renewal
sudo certbot renew --dry-run

# Add to crontab for automatic renewal
0 0 * * * certbot renew --quiet
```

---

## 📊 Production Deployment Checklist

### Pre-Deployment

- [ ] Update all demo URLs in `demo.html`
- [ ] Test all links and navigation
- [ ] Verify mobile responsiveness
- [ ] Check browser compatibility (Chrome, Firefox, Safari, Edge)
- [ ] Optimize images (compress, WebP format)
- [ ] Minify CSS/JS (optional, already optimized)
- [ ] Test live demo integration
- [ ] Configure analytics (Google Analytics, Plausible)
- [ ] Set up error monitoring (Sentry)

### Security

- [ ] Enable HTTPS/SSL
- [ ] Configure security headers (CSP, X-Frame-Options)
- [ ] Set up CORS properly
- [ ] Review and sanitize any user inputs
- [ ] Enable rate limiting on demo server
- [ ] Configure firewall rules
- [ ] Set up DDoS protection (CloudFlare)

### Performance

- [ ] Enable Gzip compression
- [ ] Configure CDN (CloudFront, CloudFlare)
- [ ] Set appropriate cache headers
- [ ] Enable HTTP/2
- [ ] Optimize font loading
- [ ] Lazy load images (if needed)
- [ ] Monitor page load times

### Monitoring

- [ ] Set up uptime monitoring (UptimeRobot, Pingdom)
- [ ] Configure log aggregation (CloudWatch, ELK)
- [ ] Set up alerts for errors
- [ ] Monitor server resources
- [ ] Track website analytics
- [ ] Monitor demo server health

---

## 🌐 DNS Configuration

### Example DNS Records

```
# Main website
A       openfinops.io           → 185.199.108.153
A       openfinops.io           → 185.199.109.153
A       openfinops.io           → 185.199.110.153
A       openfinops.io           → 185.199.111.153
CNAME   www.openfinops.io       → rdmurugan.github.io

# Demo server
A       demo.openfinops.io      → [your-server-ip]

# CDN (optional)
CNAME   cdn.openfinops.io       → d1234567890.cloudfront.net
```

---

## 🚀 Quick Start Commands

### Deploy to GitHub Pages
```bash
git add website/
git commit -m "Deploy website"
git push origin main
# Enable in Settings → Pages
```

### Deploy to Netlify
```bash
netlify deploy --prod --dir=website
```

### Deploy to Vercel
```bash
vercel --prod --cwd website
```

### Deploy with Docker
```bash
docker-compose up -d
```

---

## 📞 Support

- **GitHub Issues:** https://github.com/rdmurugan/openfinops/issues
- **Documentation:** http://localhost:8888/documentation.html
- **API Reference:** http://localhost:8888/api.html

---

## 📄 License

Apache-2.0 License - See LICENSE file in project root.
