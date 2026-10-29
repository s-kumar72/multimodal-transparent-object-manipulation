#!/usr/bin/env bash
# Download the public transparent-object datasets into $TRANSPARENT_RAW_ROOT (default ~/datasets/transparent/raw)
# and verify them. Needs: curl, gdown (pip install gdown), sha256sum. ~9.5 GB download.
#   bash real_data/download.sh            # all four
#   bash real_data/download.sh rgbt_glass # just one
# TRansPose is NOT included: it is behind a request form (see real_data/README.md).
set -euo pipefail
RAW=${TRANSPARENT_RAW_ROOT:-$HOME/datasets/transparent/raw}
WANT=${*:-rgbd_gsd mju_glass rgbt_glass trobi}

get() {  # url dest
  [ -s "$2" ] && { echo "have $2"; return; }
  mkdir -p "$(dirname "$2")"; echo "get  $2"
  curl -fL --retry 5 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}
gd() {   # google-drive-id dest
  [ -s "$2" ] && { echo "have $2"; return; }
  mkdir -p "$(dirname "$2")"; echo "get  $2"
  gdown "$1" -O "$2"
  [ -s "$2" ] || { echo "gdown failed for $2 (Drive quota? retry later or download in a browser)"; exit 1; }
}

for d in $WANT; do case $d in
  rgbd_gsd)    # 1.9 GB, HF parquet, CC BY-NC 4.0
    for f in data/test-00000-of-00001.parquet data/train-00000-of-00002.parquet data/train-00001-of-00002.parquet README.md; do
      get "https://huggingface.co/datasets/garrying/RGBD-GSD/resolve/main/$f" "$RAW/rgbd_gsd/$f"; done ;;
  mju_glass)   # 1.75 GB, GitHub LFS
    for f in images depth mask; do
      get "https://media.githubusercontent.com/media/weduake/MJU_Glass/main/dataset/$f.zip" "$RAW/mju_glass/$f.zip"; done ;;
  rgbt_glass)  # 3.2 GB zip (16 GB unpacked; the converter reads the zip directly)
    gd 1g42XPaudslnIo1oR59FKQV9rVFOTqu_w "$RAW/rgbt_glass/RGB-T-glass.zip" ;;
  trobi)       # 2.4 GB real scenes + CAD models; the 36 GB synthetic set (1V7E3vqfM1NIkeafY8JtxAlEpYyOJHn16) is skipped
    gd 1Nmxym76NCM6xd76lD2W7dBfAIclqsXWM "$RAW/trobi/TROBI.zip"
    gd 1LWPmiem2IfGrhehwkZF1OuuthD0jFNzX "$RAW/trobi/T_ROBI_Object_models.zip" ;;
  *) echo "unknown dataset $d"; exit 1 ;;
esac; done

cd "$RAW"
sha256sum -c --ignore-missing <<'SUMS'
c90d2fb04d03e81e781c1bf8fc385fe2ab378c22d745a54a477d6552a5943123  rgbd_gsd/data/test-00000-of-00001.parquet
485dfef78029141496c0a7969ef66f6a7d9c05cb91091c2a0bb764fba90a2d08  rgbd_gsd/data/train-00000-of-00002.parquet
383354afb2f44fb4c6356f9ff979a24acf4e58849417aac5db8579ac41041a55  rgbd_gsd/data/train-00001-of-00002.parquet
7443f459d7cffd5bd0e31a418cc582e42905941f31e0af44d1ae4f09df51e918  mju_glass/images.zip
e614d12cf13c1ba15969053c6eacc6c7088d99553eaf9d0afddacb0161197676  mju_glass/depth.zip
9ffb0e4f2e8612e4fe68ee82718d246ac00898ae1fa28b6a877ff0b1e3064cdf  mju_glass/mask.zip
8c919e359678376b43f44e6f25121c5404f33231f222d84805b4720f2d4d9191  rgbt_glass/RGB-T-glass.zip
34cdfc997a110f05b453778b783eb0d6f739478b8a59aa29e218597767a8da29  trobi/TROBI.zip
e46096315e479cc3ab8e87910469cbe7a91d9639e089ab302c8e601c456e4152  trobi/T_ROBI_Object_models.zip
SUMS
echo "done -> $RAW"
