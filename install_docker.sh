#!/bin/bash
set -e

# Purpose: End-user installation script for Morphik
# This script downloads docker-compose.run.yml, creates .env file, downloads morphik.toml,
# and starts all services. Creates start-morphik.sh for easy restarts with automatic
# port detection from morphik.toml configuration.
# Usage: curl -sSL https://install.morphik.ai | bash

# --- Configuration ---
REPO_URL="https://raw.githubusercontent.com/morphik-org/morphik-core/main"
COMPOSE_FILE="docker-compose.run.yml"
DIRECT_INSTALL_URL="https://www.morphik.ai/docs/getting-started#self-host-direct-installation-advanced"

# --- Helper Functions ---
print_info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

print_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1" >&2
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "'$1' is not installed. Please install it to continue."
    fi
}

# --- Main Script ---

# 1. Check for prerequisites
print_info "Checking for Docker and Docker Compose..."
check_command "docker"
if ! docker compose version &> /dev/null; then
    print_error "Docker Compose V2 is required. Please ensure it's installed and accessible."
fi
print_success "Prerequisites are satisfied."

# 2. Apple Silicon Warning
if [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    print_warning "You are on an Apple Silicon Mac (arm64)."
    print_warning "For best performance (including GPU access), we strongly recommend the Direct Installation method."
    print_warning "You can find the guide here: $DIRECT_INSTALL_URL"
    read -p "Do you want to continue with the Docker installation anyway? (y/N): " choice < /dev/tty
    if [[ "$choice" != "y" && "$choice" != "Y" ]]; then
        echo "Installation aborted by user."
        exit 0
    fi
fi

# 3. Download necessary files
print_info "Downloading the Docker Compose configuration file..."
if curl -fsSL -o "$COMPOSE_FILE" "$REPO_URL/$COMPOSE_FILE"; then
    print_success "Downloaded '$COMPOSE_FILE'."
else
    print_error "Failed to download '$COMPOSE_FILE'. Please check your internet connection and the repository URL."
fi

# 4. Create .env and get User Input for API Key
print_info "Creating '.env' file for your secrets..."
cat > .env <<EOF
# Your OpenAI API key is required
OPENAI_API_KEY=

# A secret key for signing JWTs. A random one is generated for you.
JWT_SECRET_KEY=your-super-secret-key-that-is-long-and-random-$(openssl rand -hex 16)
EOF

read -p "Please enter your OpenAI API Key (it will be saved to the .env file): " openai_api_key < /dev/tty
if [[ -z "$openai_api_key" ]]; then
    print_error "OpenAI API Key cannot be empty."
fi

# Use sed to safely replace the key in the .env file.
sed -i.bak "s|OPENAI_API_KEY=|OPENAI_API_KEY=$openai_api_key|" .env
rm -f .env.bak
print_success "'.env' file has been configured with your API key."

# 5. Download and setup configuration
print_info "Setting up configuration file..."
print_info "Extracting default 'morphik.toml' for you to customize..."
docker run --rm ghcr.io/morphik-org/morphik-core:latest \
       cat /app/morphik.toml.default > morphik.toml

print_info "Enabling configuration mounting in '$COMPOSE_FILE'..."
# Use sed to uncomment the volume mount lines for both services
sed -i.bak 's|# - ./morphik.toml:/app/morphik.toml:ro|- ./morphik.toml:/app/morphik.toml:ro|g' "$COMPOSE_FILE"
rm -f ${COMPOSE_FILE}.bak

print_success "Configuration has been set up at 'morphik.toml'."
print_info "You can edit this file to customize models, ports, or other settings."
read -p "Press [Enter] to continue with the current configuration or edit 'morphik.toml' in another terminal first..." < /dev/tty

# Update port mapping in docker-compose.run.yml to match morphik.toml
API_PORT=$(awk '/^\[api\]/{flag=1; next} /^\[/{flag=0} flag && /^port[[:space:]]*=/ {gsub(/^port[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || echo "8000")
sed -i.bak "s|\"8000:8000\"|\"${API_PORT}:${API_PORT}\"|g" "$COMPOSE_FILE"
rm -f ${COMPOSE_FILE}.bak

# 5.5. Ask about UI installation
echo ""
print_info "Morphik includes an admin UI for easier interaction."
read -p "Would you like to install the Admin UI? (y/N): " install_ui < /dev/tty

UI_PROFILE=""
if [[ "$install_ui" == "y" || "$install_ui" == "Y" ]]; then
    print_info "Extracting UI component files from Docker image..."
    # Extract the UI component from the Docker image (now included in the image)
    docker run --rm ghcr.io/morphik-org/morphik-core:latest \
           tar -czf - -C /app ee/ui-component | tar -xzf -

    if [ -d "ee/ui-component" ]; then
        print_success "UI component downloaded successfully."
        UI_PROFILE="--profile ui"

        # Update NEXT_PUBLIC_API_URL to use the correct port
        sed -i.bak "s|NEXT_PUBLIC_API_URL=http://localhost:8000|NEXT_PUBLIC_API_URL=http://localhost:${API_PORT}|g" "$COMPOSE_FILE"
        rm -f ${COMPOSE_FILE}.bak

        # Save UI installation flag for start-morphik.sh
        echo "UI_INSTALLED=true" >> .env
    else
        print_warning "Failed to download UI component. Continuing without UI."
    fi
fi

# 6. Start the application
print_info "Starting the Morphik stack... This may take a few minutes for the first run."
docker compose -f "$COMPOSE_FILE" $UI_PROFILE up -d

print_success "🚀 Morphik has been started!"
print_info "📝 Check the logs for status - it can take a few minutes to fully load"
print_info "🔄 The URL will show 'unavailable' until the service is ready"

# Read port from morphik.toml to display correct URL
API_PORT=$(awk '/^\[api\]/{flag=1; next} /^\[/{flag=0} flag && /^port[[:space:]]*=/ {gsub(/^port[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || echo "8000")

echo ""
print_info "🌐 API endpoints:"
print_info "   Health check: http://localhost:${API_PORT}/health"
print_info "   API docs:     http://localhost:${API_PORT}/docs"
print_info "   Main API:     http://localhost:${API_PORT}"

if [[ -n "$UI_PROFILE" ]]; then
    echo ""
    print_info "🎨 Admin UI:"
    print_info "   Interface:    http://localhost:3003"
    print_info "   Note: The UI may take a few minutes to build on first run"
fi

echo ""
print_info "📋 Management commands:"
print_info "   View logs:    docker compose -f $COMPOSE_FILE $UI_PROFILE logs -f"
print_info "   Stop services: docker compose -f $COMPOSE_FILE $UI_PROFILE down"
print_info "   Restart:      ./start-morphik.sh"

# Create convenience startup script
cat > start-morphik.sh << 'EOF'
#!/bin/bash
set -e

# Purpose: Production startup script for Morphik
# Automatically updates port mapping from morphik.toml and includes UI if installed

API_PORT=$(awk '/^\[api\]/{flag=1; next} /^\[/{flag=0} flag && /^port[[:space:]]*=/ {gsub(/^port[[:space:]]*=[[:space:]]*/, ""); print; exit}' morphik.toml 2>/dev/null || echo "8000")
CURRENT_PORT=$(grep -oE '"[0-9]+:[0-9]+"' docker-compose.run.yml | head -1 | cut -d: -f1 | tr -d '"')

if [ "$CURRENT_PORT" != "$API_PORT" ]; then
    echo "Updating port mapping from $CURRENT_PORT to $API_PORT..."
    sed -i.bak "s|\"${CURRENT_PORT}:${CURRENT_PORT}\"|\"${API_PORT}:${API_PORT}\"|g" docker-compose.run.yml
    rm -f docker-compose.run.yml.bak
fi

# Check if UI is installed
UI_PROFILE=""
if [ -f ".env" ] && grep -q "UI_INSTALLED=true" .env; then
    UI_PROFILE="--profile ui"
fi

docker compose -f docker-compose.run.yml $UI_PROFILE up -d
echo "🚀 Morphik is running on http://localhost:${API_PORT}"
echo "   Health: http://localhost:${API_PORT}/health"
echo "   Docs:   http://localhost:${API_PORT}/docs"
if [ -n "$UI_PROFILE" ]; then
    echo ""
    echo "🎨 Admin UI: http://localhost:3003"
fi
EOF
chmod +x start-morphik.sh

echo ""
print_success "🎉 Enjoy using Morphik!"
