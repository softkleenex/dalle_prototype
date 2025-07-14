mkdir /content/CAH
cd /content/CAH
wget --continue https://the-eye.eu/public/AI/cah/laion400m-dat-release.torrent
aria2c --continue --seed-time=0 --file-allocation='trunc' --select-file=1-100 laion400m-dat-release.torrent
cd /content
