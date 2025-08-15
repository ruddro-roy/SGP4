# Static site Dockerfile
FROM nginx:alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Copy static files
COPY . /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create non-root user
RUN addgroup -g 1001 -S static && \
    adduser -S static -u 1001

# Set proper permissions
RUN chown -R static:static /usr/share/nginx/html && \
    chown -R static:static /var/cache/nginx && \
    chown -R static:static /var/log/nginx && \
    chown -R static:static /etc/nginx/conf.d

RUN touch /var/run/nginx.pid && \
    chown -R static:static /var/run/nginx.pid

# Switch to non-root user
USER static

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]