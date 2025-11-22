#!/bin/bash

# Create necessary directories
mkdir -p certbot/www certbot/conf

# Stop nginx if it's running
echo "Stopping nginx..."
docker compose stop ethercheck-nginx

# Generate SSL certificate using certbot in standalone mode
echo "Generating SSL certificate..."
docker compose run --rm --service-ports certbot certonly \
  --standalone \
  --email slavapmk@gmail.com \
  --agree-tos \
  --no-eff-email \
  -d monitor.slavapmk.ru

if [ $? -ne 0 ]; then
    echo "Failed to generate SSL certificate with standalone method, trying with host network..."
    # Update the certbot service to use host network temporarily
    sed -i 's/^  certbot:.*/  certbot:\n    image: certbot\/certbot:latest\n    network_mode: host\n    volumes:\n      - .\/certbot\/www:\/var\/www\/certbot:rw\n      - .\/certbot\/conf:\/etc\/letsencrypt:rw\n    command: echo "Certbot container ready"/' docker-compose.yaml

    docker compose run --rm certbot certonly \
      --standalone \
      --email slavapmk@gmail.com \
      --agree-tos \
      --no-eff-email \
      -d monitor.slavapmk.ru

    # Revert the change to the docker-compose.yaml
    sed -i 's/^  certbot:.*/  certbot:\n    image: certbot\/certbot:latest\n    volumes:\n      - .\/certbot\/www:\/var\/www\/certbot:rw\n      - .\/certbot\/conf:\/etc\/letsencrypt:rw\n    depends_on:\n      - ethercheck-nginx\n    command: echo "Certbot container ready"/' docker-compose.yaml
fi

# Check if certificate was generated successfully
if [ -f "certbot/conf/live/monitor.slavapmk.ru/fullchain.pem" ]; then
    echo "SSL certificate generated successfully!"
    echo "Starting nginx with SSL configuration..."
    docker compose up -d ethercheck-nginx
else
    echo "Failed to generate SSL certificate"
    echo "Check the logs above for details"
    exit 1
fi
