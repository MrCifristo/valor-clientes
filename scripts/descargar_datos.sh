#!/usr/bin/env bash
# Descarga el dataset completo (7 CSV, ~520 MB) desde el Release del repositorio
# y lo deja en data/01_raw/. Requiere curl y unzip.
set -euo pipefail

URL="https://github.com/MrCifristo/valor-clientes/releases/download/datos-v1/datos_completos.zip"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"

curl -L --fail -o "$RAIZ/datos_completos.zip" "$URL"
unzip -o "$RAIZ/datos_completos.zip" -d "$RAIZ/data/01_raw"
rm "$RAIZ/datos_completos.zip"
ls -lh "$RAIZ/data/01_raw"/*.csv
