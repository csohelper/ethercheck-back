#!/bin/bash

docker compose run --rm certbot certonly --webroot --webroot-path /var/www/certbot/ -d monitor.slavapmk.ru

# https://dev.to/mindsers/https-using-nginx-and-lets-encrypt-in-docker-1ama
