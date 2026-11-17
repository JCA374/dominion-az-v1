#!/usr/bin/env bash
# Narrate, listen-check, export, publish podcast episode NN to the project's GitHub Pages site, and verify it.
# Run prepare.sh NN first. Ends with "DEPLOYED <url>" or "FAILED: <reason>".
set -uo pipefail
. ~/.claude/skills/project-podcast/scripts/conf.sh
# narration with this project's own Python 3.12 venv (.voice-venv: kokoro, faster-whisper, torch CPU)
V="$PWD/.voice-venv/bin/python $HOME/.claude/skills/voice-text/scripts/voice.py"
NN=${1:?usage: publish.sh NN}
D=voice/$SLUG_PREFIX-$NN
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "FAILED: $*"; exit 1; }
[ -d "$D" ] || fail "run prepare.sh $NN first"
[ -f docs/podcast/index.html ] || { mkdir -p docs/podcast; cp $SK/assets/index.html docs/podcast/index.html; log "added docs/podcast/index.html"; }

log "prep + narrate (Kokoro)"
$V prep $D > /dev/null 2>&1 || fail "prep"
nice -n 5 $V narrate $D > $D/narrate.log 2>&1 || fail "narrate (see $D/narrate.log)"
log "listen-check (Whisper)"
$V listen $D > $D/listen.txt 2>&1 || true
$V export $D > $D/export.log 2>&1 || fail "export (see $D/export.log)"
log "pipeline stages PASS: $($V status $D 2>&1 | grep -c PASS)/5"
grep -E "swapped|missing|extra|garbled" $D/listen.txt | sed 's/^/  listen flag: /'

cp $D/audio/ch01.mp3 docs/podcast/episode-$NN.mp3
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 docs/podcast/episode-$NN.mp3)
log "episode-$NN.mp3: ${dur%.*} s"
python3 $SK/scripts/make_feed.py || fail "make_feed"
if [ -n "$TESTS" ]; then log "tests"; eval "$TESTS" 2>&1 | tail -1; fi

git add docs/podcast/index.html docs/podcast/episode-$NN.mp3 docs/data/podcast.json podcast/episode-$NN.md podcast/episode-$NN.notes.json 2>/dev/null
title=$(head -1 podcast/episode-$NN.md | sed -e 's/^# //' -e 's/^Episode [0-9]*: //')
git commit -q -m "Podcast: $title

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || log "nothing new to commit"
git push -q origin HEAD || fail "git push"
sha=$(git rev-parse HEAD)

log "waiting for GitHub Pages ($sha)"
b=""
for i in $(seq 1 60); do
  b=$(gh api repos/$REPO/pages/builds/latest --jq '.status+" "+.commit' 2>/dev/null)
  [ "$b" = "built $sha" ] && break
  sleep 10
done
[ "$b" = "built $sha" ] || log "warning: latest Pages build is '$b'"
sleep 15
for u in podcast/index.html podcast/episode-$NN.mp3 data/podcast.json; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$SITE/$u?v=$sha")
  log "  $code $SITE/$u"
  [ "$code" = 200 ] || fail "$u returned $code"
done
# The Pages CDN can serve the old feed for a minute or two after a build: retry for up to 5 minutes.
listed=0
for i in $(seq 1 30); do
  if curl -s "$SITE/data/podcast.json?v=$sha-$i" | grep -q "episode-$NN.mp3"; then listed=1; break; fi
  sleep 10
done
[ $listed -eq 1 ] || fail "feed doesn't list episode $NN after 5 minutes"
echo "DEPLOYED $SITE/podcast/index.html (episode $NN: $title, ${dur%.*} s)"
