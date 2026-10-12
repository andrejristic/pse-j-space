#!/bin/sh
# Pravi pse-a_<verzija>_all.deb. Pokrece se na Acer-u; paket radi na bilo kom
# x86-64 ili arm64 Ubuntu/Debian sistemu sa Python >= 3.10.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
V=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$HERE/pse_a/__init__.py")
PKG="pse-a_${V}_all"
STAGE="$(mktemp -d)/$PKG"
mkdir -p "$STAGE/DEBIAN" "$STAGE/usr/lib/pse-a" "$STAGE/usr/bin" \
         "$STAGE/usr/share/applications" "$STAGE/usr/share/doc/pse-a"

# izvorni kod, bez kesa
(cd "$HERE" && find pse_a -type f \( -name '*.py' -o -name '*.html' -o -name '*.css' \
      -o -name '*.js' -o -name '*.txt' \) ! -path '*/__pycache__/*' -print) | while read -r f; do
    install -D -m 0644 "$HERE/$f" "$STAGE/usr/lib/pse-a/$f"
done
install -m 0644 "$HERE/packaging/requirements.txt" "$STAGE/usr/lib/pse-a/requirements.txt"
install -m 0755 "$HERE/packaging/pse-a" "$STAGE/usr/bin/pse-a"
install -m 0644 "$HERE/packaging/pse-a.desktop" "$STAGE/usr/share/applications/pse-a.desktop"
install -m 0644 "$HERE/LICENSE" "$STAGE/usr/share/doc/pse-a/LICENSE"
install -m 0644 "$HERE/NOTICE" "$STAGE/usr/share/doc/pse-a/NOTICE"
for d in PSE-A_MASTER_ALGORITHM.docx PSE-A_PLAN.docx PSE-A_DOWNLOAD_LINKS.docx PSE-A_UPUTSTVO.docx; do
    [ -f "$HERE/$d" ] && install -m 0644 "$HERE/$d" "$STAGE/usr/share/doc/pse-a/$d"
done

SIZE=$(du -sk "$STAGE/usr" | cut -f1)
cat > "$STAGE/DEBIAN/control" <<CTL
Package: pse-a
Version: $V
Architecture: all
Maintainer: Andrej Ristic <andrej.s.ristic@gmail.com>
Installed-Size: $SIZE
Depends: python3 (>= 3.10), python3-venv
Recommends: chromium | firefox
Section: science
Priority: optional
Description: PSE Analyzer - spektralna analiza lokalnih jezickih modela
 Otvara GGUF model, meri spektar svake matrice tezina (Marchenko-Pastur,
 alfa teskog repa, izdvojeni pravci), poredi preciznosti istog modela i
 cuva zapecacene rezultate u dosijeu. Web pogled samo na 127.0.0.1.
 .
 Paket ne sadrzi kompajliran kod. numpy/scipy se instaliraju u korisnicki
 venv pri prvom pokretanju, na masini na kojoj ce raditi.
CTL
cat > "$STAGE/DEBIAN/postinst" <<'PI'
#!/bin/sh
set -e
echo "pse-a instaliran. Prvo pokretanje pravi okruzenje (treba internet):"
echo "    pse-a doctor      -- sta ova masina moze i koliko brzo"
echo "    pse-a selftest    -- invarijante; sa --k2 <folder> i dequantizacija"
echo "Podaci ostaju u ~/.local/share/pse-a i posle deinstalacije."
PI
chmod 0755 "$STAGE/DEBIAN/postinst"

mkdir -p "$HERE/dist"
dpkg-deb --root-owner-group -Zxz --build "$STAGE" "$HERE/dist/$PKG.deb" >/dev/null
echo "$HERE/dist/$PKG.deb"
