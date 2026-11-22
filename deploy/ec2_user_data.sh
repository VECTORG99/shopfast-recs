#!/bin/bash
# User data para Amazon Linux 2023: instala Docker, Compose, Git, Python y make
dnf update -y
dnf install -y docker git python3.11 python3.11-pip make
systemctl enable --now docker
usermod -aG docker ec2-user
mkdir -p /usr/local/lib/docker/cli-plugins
curl -sSL "https://github.com/docker/compose/releases/download/v2.30.3/docker-compose-linux-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
