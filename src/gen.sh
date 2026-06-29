#!/usr/bin/env bash
# Generate the 10 Vitrine infographic slides via nano_banana (Gemini image API).
# Usage: bash gen.sh [model] [size] [only_keys...]
#   bash gen.sh flash 1K            # fast low-res pass, all 10
#   bash gen.sh pro 2K              # final hi-res pass, all 10
#   bash gen.sh pro 2K 07 08        # regenerate only slides 07 and 08
set -u
MODEL="${1:-flash}"; SIZE="${2:-1K}"; shift || true; shift || true
ONLY=("$@")
OUT=/tmp/deck2/out; mkdir -p "$OUT" /tmp/deck2/log
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$GOOGLE_GEMINI_API_KEY}"
NB=/home/devuser/workspace/gaussian/LichtFeld-Studio/report/poster/nano_banana.py

STYLE="A world-class presentation infographic, one slide from a single cohesive 10-slide deck with identical art direction. ART STYLE: soft 3D isometric illustration with clean vector clarity; dark midnight navy-to-teal gradient background; luminous cyan and electric-blue glows with a warm amber accent; a points-of-light particle-cloud motif throughout (evoking a cloud of glowing dots); cinematic soft top-left lighting; gentle depth, bokeh and subtle reflections; premium Apple-keynote feel; generous negative space, uncluttered and elegant; universally readable pictogram icons. FORMAT: 16:9 landscape, one clear centered focal subject. TEXT RULE: minimize text; render only the few exact words specified (if any), spelled correctly and large; never invent extra letters, captions, labels, logos or watermarks."

declare -A P
P[01]="$STYLE SCENE: on the left a sleek smartphone films a cozy real living room; from its lens a flowing stream of glowing cyan light-particles sweeps to the right and reassembles into a luminous floating 3D holographic model of that same room. A feeling of magic and transformation, lots of depth. The single word VITRINE in clean large modern capital letters near the top, and no other text."
P[02]="$STYLE SCENE: a beautiful historic heritage interior (a museum-like room with statues and artifacts) being gently scanned: a soft sweeping grid of glowing cyan points passes over the furniture and walls and lifts a translucent glowing 3D twin of the room into the air above. Warm amber light on the real heritage objects, cool cyan on the digital twin. Feeling: preserving real places forever as living 3D. No text."
P[03]="$STYLE SCENE: a split composition. On the left, a messy leaning stack of shaky, blurry, dark video frames (motion blur, imperfect). On the right, one crisp clean glowing 3D model of a room. Between them a luminous gap or chasm being crossed by a half-built bridge of light. Conveys: ordinary video is messy and turning it into good 3D is the hard part. No text."
P[04]="$STYLE SCENE: an exploded view: on the left a glowing 3D room whose furniture lifts out as a few clearly separate, individually selectable 3D object pieces (a chair, a cabinet, a tool case), each with its own soft glowing outline and a small friendly green check mark, gliding to the right INTO a sleek glowing game-engine screen/portal where they land as ready-to-use assets. Conveys: you get a reusable room PLUS separate ready-made object game-assets. No text."
P[05]="$STYLE SCENE: an incoming video clip enters a smart glowing decision dial/gauge that reads its quality; from the dial two or three clearly different glowing routes branch forward like selectable roads, and ONE route lights up brightest as the chosen best path while the others dim. A clear sense of an intelligent automatic choice. No text."
P[06]="$STYLE SCENE: a clean left-to-right luminous conveyor flow with five evenly spaced glowing 3D icons joined by bright forward arrows: (1) a video camera, (2) a cloud of glowing points, (3) a smooth grey 3D shape, (4) a colourful textured 3D object, (5) a glowing game-engine screen-cube. Simple, balanced, each step clearly leading to the next. Tiny step numbers 1 2 3 4 5 under the icons and no other text."
P[07]="$STYLE SCENE: a friendly capable robot AI assistant standing inside a glowing translucent glass container (a shipping container made of light), confidently operating a whole miniature automated factory of pipeline machines around it with several helpful robotic arms; bright circular arrows around it show a loop of diagnose, run, check, fix. It looks autonomous, calm and in control, orchestrating everything by itself. Warm, optimistic, impressive hero shot. No text."
P[08]="$STYLE SCENE: a clean celebratory data dashboard floating in space with three big glowing visuals: a semicircular quality gauge/dial sweeping up into the green, a large upward arrow beside the bold number +38%, and a small rising bar chart next to a neat cluster of glowing 3D object-cubes. Bright, simple, motivating. Show only these exact characters large and correct: +38% and 4K. No other text."
P[09]="$STYLE SCENE: an elegant floating toolbox / control panel with a row of interchangeable glowing module cards in slots, each card showing a different small 3D method icon (a particle splat cloud, a wireframe mesh, a textured object chip), with one card lifted and highlighted by a glowing selection ring as if being chosen. Conveys: a flexible menu of tools, pick the right one. No text."
P[10]="$STYLE SCENE: a breathtaking finished 3D scene of the reconstructed room displayed huge and cinematic inside a glowing game-engine screen; a person wearing a sleek lightweight AR/VR headset reaches out to explore the floating world, full of wonder; a hopeful sunrise glow of warm amber blending into cyan. Conveys: the finished living world, ready to explore, the future. No text."

gen(){ local k="$1"; python3 "$NB" --model "$MODEL" --size "$SIZE" --aspect 16:9 \
  --prompt "${P[$k]}" --output "$OUT/s$k.png" >"/tmp/deck2/log/$k.txt" 2>&1 \
  && echo "  ok  s$k ($MODEL/$SIZE)" || echo "  FAIL s$k -> $(tail -1 /tmp/deck2/log/$k.txt)"; }

KEYS=("${ONLY[@]}"); [ ${#KEYS[@]} -eq 0 ] && KEYS=(01 02 03 04 05 06 07 08 09 10)
echo "generating: ${KEYS[*]}  ($MODEL/$SIZE)"
i=0
for k in "${KEYS[@]}"; do gen "$k" & i=$((i+1)); if (( i % 3 == 0 )); then wait; fi; done
wait
echo "== results =="; ls -la "$OUT" | grep -E 's[0-9]+\.png' || true
