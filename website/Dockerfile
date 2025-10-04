FROM nginx:alpine

# Metadata
LABEL maintainer="OpenFinOps Team"
LABEL description="OpenFinOps Marketing Website"
LABEL version="1.0.0"

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy website files
COPY index.html /usr/share/nginx/html/
COPY features.html /usr/share/nginx/html/
COPY documentation.html /usr/share/nginx/html/
COPY api.html /usr/share/nginx/html/
COPY demo.html /usr/share/nginx/html/
COPY css/ /usr/share/nginx/html/css/
COPY js/ /usr/share/nginx/html/js/
COPY images/ /usr/share/nginx/html/images/

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Remove default nginx config
RUN rm -f /etc/nginx/conf.d/default.conf.bak

# Set proper permissions
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# Expose ports
EXPOSE 80 443

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
