#!/usr/bin/env bash
# Install official AWS CLI v2 for current user (no sudo). Fixes Homebrew pyexpat errors on Mac.
set -euo pipefail

HOME_DIR="${HOME}"
INSTALL_ROOT="${HOME_DIR}/aws-cli"
BIN_DIR="${HOME_DIR}/.local/bin"
CHOICES_FILE="$(mktemp /tmp/aws-cli-choices.XXXXXX.xml)"

cat > "${CHOICES_FILE}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <array>
    <dict>
      <key>choiceAttribute</key>
      <string>customLocation</string>
      <key>attributeSetting</key>
      <string>${INSTALL_ROOT}</string>
      <key>choiceIdentifier</key>
      <string>default</string>
    </dict>
  </array>
</plist>
EOF

mkdir -p "${INSTALL_ROOT}" "${BIN_DIR}"
curl -fsSL "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o /tmp/AWSCLIV2.pkg
installer -pkg /tmp/AWSCLIV2.pkg -target CurrentUserHomeDirectory -applyChoiceChangesXML "${CHOICES_FILE}"
ln -sf "${INSTALL_ROOT}/aws-cli/aws" "${BIN_DIR}/aws"
ln -sf "${INSTALL_ROOT}/aws-cli/aws_completer" "${BIN_DIR}/aws_completer"
rm -f "${CHOICES_FILE}"

if ! grep -q '.local/bin' "${HOME_DIR}/.zshrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME_DIR}/.zshrc"
    echo "Added ~/.local/bin to PATH in ~/.zshrc"
fi

echo "Installed: $("${BIN_DIR}/aws" --version)"
echo "Run: source ~/.zshrc && aws configure --profile gnnplus"
