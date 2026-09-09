#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${repo_root}/media/downloads/demo_dog.jpg"
expected_sha256="4ba47c188c223b064a9ef242acda35f308a8402a650bfa2389cf4a556daa01fd"
url="https://upload.wikimedia.org/wikipedia/commons/2/2a/Bully-dog-2314909-1920.jpg"

mkdir -p "$(dirname "${destination}")"
if [[ ! -f "${destination}" ]]; then
  curl --fail --location --show-error "${url}" --output "${destination}"
fi

actual_sha256="$(sha256sum "${destination}" | cut -d ' ' -f 1)"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "Checksum mismatch for ${destination}" >&2
  echo "Expected: ${expected_sha256}" >&2
  echo "Actual:   ${actual_sha256}" >&2
  exit 1
fi

echo "Ready: ${destination}"

