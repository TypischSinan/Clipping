# Clipper

YouTube-Link rein, fertige vertikale Clips raus. Läuft komplett lokal auf der eigenen GPU.

```bash
clipper analyze "https://www.youtube.com/watch?v=..."
clipper select <video-id> --from clips.json
clipper build <video-id>
```

Aus einem 20-Minuten-Video entstehen so 30+ eigenständige 9:16-Clips mit eingebrannten
Untertiteln, Hook-Overlay, motivzentriertem Crop und fertigen TikTok-Captions.

> Die Kommentare im Code und diese Dokumentation sind auf Deutsch, die erzeugten
> Clips standardmäßig auf Englisch (`select.output_language`).

---

## Inhalt

- [Wofür das gebaut ist](#wofür-das-gebaut-ist)
- [Was die Pipeline macht](#was-die-pipeline-macht)
- [Installation](#installation)
- [Die zwei Wege durch die Pipeline](#die-zwei-wege-durch-die-pipeline)
- [Kommandoreferenz](#kommandoreferenz)
- [Konfiguration](#konfiguration)
- [Design-Entscheidungen](#design-entscheidungen)
- [Maximale Ausbeute pro Video](#maximale-ausbeute-pro-video)
- [Untertitel](#untertitel)
- [Bekannte Grenzen](#bekannte-grenzen)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [Projektstruktur](#projektstruktur)
- [Rechtliches](#rechtliches)

---

## Wofür das gebaut ist

Clipping-Kampagnen (Whop / Content Rewards, Vyro, clipping.net) zahlen pro 1000 Views
für kurze vertikale Clips aus fremdem Long-Form-Material. Der Engpass ist nicht die
Reichweite, sondern der Durchsatz: aus jedem Quellvideo möglichst viele brauchbare
Clips zu bekommen, ohne stundenlang selbst zu schneiden.

Genau das macht dieses Projekt. Es ist ein lokales, selbst gehostetes Äquivalent zu
OpusClip — mit dem Unterschied, dass die Momentauswahl von einem Modell mit
**visuellem Verständnis** kommt und nicht nur aus dem Transkript.

Der Unterschied ist bei Action- und Challenge-Content entscheidend. Bei einem Podcast
steht im Transkript, wo der gute Moment ist. Bei MrBeast steht er da nicht: „oh my god"
sagt nichts darüber aus, ob gerade ein Auto explodiert oder jemand eine Tür öffnet.
Deshalb gehen Keyframes als Bilder mit in die Auswahl.

---

## Was die Pipeline macht

| # | Stufe | Werkzeug | Ergebnis |
|---|-------|----------|----------|
| 1 | Download | yt-dlp | `work/<id>/source.mp4` |
| 2 | Transkript | faster-whisper (CUDA) | Wort-genaue Zeitstempel |
| 3 | Shot-Erkennung | PySceneDetect | Schnittgrenzen |
| 4 | Audio-Energie | ffmpeg + numpy | Lautstärke-Hüllkurve, Peaks |
| 5 | Momentauswahl | Claude Opus 5 *oder* Claude-Code-Session | Clips mit Hook, Caption, Score |
| 6 | Reframe | OpenCV / YuNet | 9:16 mit Motivverfolgung |
| 7 | Untertitel | ASS | Wort-für-Wort-Hervorhebung + Hook |
| 8 | Rendern | ffmpeg | `out/<id>/NNN_score_titel.mp4` |

Jede Stufe cached ihr Ergebnis in `work/<video_id>/`. Ein zweiter Lauf überspringt
Download, Transkription, Shot-Erkennung und Audio-Analyse und beginnt direkt bei der
Auswahl. Das ist wichtig, weil man die Auswahl oft mehrmals anpasst.

**Ausgabe pro Video:**

```
out/<video-id>/
  001_088_Chicken_Named_Brian.mp4     # Reihenfolge_Score_Titel
  002_085_He_Tried_To_Eat_It.mp4
  ...
  clips.json                          # Hooks, Captions, Zeitstempel, Scores
```

`clips.json` ist das, was du beim Hochladen brauchst: pro Clip die fertige
TikTok-Caption inklusive Hashtags, der Hook-Text und die Quellzeitstempel.

---

## Installation

### Voraussetzungen

| | Minimum | Empfohlen |
|---|---|---|
| Python | 3.12 | 3.12 (nicht 3.13+, wegen CTranslate2-Wheels) |
| ffmpeg | im PATH | 7.x oder neuer |
| GPU | keine (CPU-Fallback) | NVIDIA mit ≥6 GB VRAM |
| RAM | 8 GB | 16 GB+ |
| Speicher | ~2 GB pro Stunde Quellvideo | |

ffmpeg unter Windows:

```bash
winget install Gyan.FFmpeg
```

### Setup

```bash
git clone https://github.com/TypischSinan/Clipping.git
cd Clipping
uv sync --extra cuda
uv pip install -e .
```

Ohne NVIDIA-GPU reicht `uv sync` ohne `--extra cuda`; Whisper läuft dann auf der CPU
(int8, deutlich langsamer, aber funktionsfähig).

Das YuNet-Modell für die Gesichtserkennung (~340 KB) lädt sich beim ersten Lauf
automatisch nach `models/`.

### Prüfen

```bash
clipper --help
python -m pytest tests/ -q
```

---

## Die zwei Wege durch die Pipeline

Die Momentauswahl (Stufe 5) ist der einzige Schritt, der ein Sprachmodell braucht.
Deshalb ist die Pipeline dort geteilt.

### Weg A — ohne API-Key, Auswahl in einer Claude-Code-Session

Der empfohlene Weg für den Eigengebrauch. Es fallen keine API-Kosten an, weil die
Auswahl in der laufenden Session passiert.

```bash
clipper analyze "https://www.youtube.com/watch?v=..." --min 15 --max 40
```

Das schreibt `work/<id>/brief.md` — ein kompaktes Briefing mit Transkript samt
Zeitstempeln, Shot-Grenzen, Energiekurve und den Pfaden der extrahierten Keyframes.
Diese Datei und die Bilder liest Claude Code, wählt die Momente und schreibt sie als
JSON:

```json
{"clips": [
  {"start": 360.5, "end": 383.0,
   "title": "Chicken Named Brian",
   "hook": "They waited 3 hours for this",
   "caption": "Three hours of waiting for one chicken 😭 #mrbeast #survival",
   "reason": "Vollstaendiger Bogen mit Pointe im letzten Satz.",
   "score": 88}
]}
```

Dann einspielen und rendern:

```bash
clipper select <id> --from clips.json
clipper build <id>
```

`select` schickt die Vorschläge durch **dieselbe** Nachbearbeitung wie den API-Pfad:
Shot-Grenzen, Wortgrenzen, Längenkorrektur, Überlappungsauflösung. Von Hand gewählte
Clips bekommen also exakt dieselben Garantien.

> Die Credentials einer Claude-Code-Session stehen Unterprozessen **nicht** zur
> Verfügung — ein Skript kann sie nicht mitbenutzen. Deshalb der geteilte Weg.

### Weg B — mit API-Key, alles in einem Lauf

```bash
$env:ANTHROPIC_API_KEY = "sk-ant-..."
clipper run "https://www.youtube.com/watch?v=..."
```

Kosten pro Video: das Briefing sind grob 20–30k Input-Tokens, dazu 24 Keyframes zu
je ~750 Tokens. Mit `--vision 0` läuft die Auswahl ohne Bilder (billiger, bei reinem
Interview-Material fast gleichwertig, bei Action-Content deutlich schlechter).

### Weg C — Heuristik, für Tests

Ohne Key fällt `run` automatisch auf eine rein akustische Auswahl zurück: Fenster um
die stärksten Energie-Spitzen. Das ist zum Durchtesten der Pipeline gedacht, nicht für
echte Clips — die Heuristik kennt nur Lautstärke und vergibt deshalb auch keine
sinnvollen Scores und keine Hooks.

---

## Kommandoreferenz

### `clipper analyze <url>`

Stufen 1–4 plus Keyframes, schreibt das Briefing. Braucht keinen API-Key.

| Option | Bedeutung |
|---|---|
| `--clips` / `-n` | Ziel-Clipzahl. `0` (Standard) = so viele wie möglich |
| `--min` / `--max` | Clip-Länge in Sekunden (Standard 15–60) |
| `--lang` | Sprache erzwingen, z. B. `en`, `de`. Sonst automatisch |
| `--whisper` | Whisper-Modell: `tiny`…`large-v3` (Standard `large-v3`) |
| `--vision` | Anzahl Keyframes fürs Briefing, `0` = keine |
| `--force` | Alle Caches verwerfen und neu rechnen |
| `-c` / `--config` | Eigene YAML über die Defaults legen |

### `clipper select <video-id> --from <datei|->`

Nimmt Clipvorschläge entgegen, räumt sie auf, schreibt `candidates.json`.
`--from -` liest von stdin. Akzeptiert `{"clips": [...]}` oder direkt eine Liste.

Gibt aus, welche Clips übernommen wurden und **welche verworfen wurden samt Grund** —
beim Maximieren der Ausbeute ist genau das die handlungsrelevante Information.

Unterstützt `--min`, `--max`, `--clips`, `-c`.

### `clipper build <video-id>`

Stufen 6–8 aus der gespeicherten Auswahl. `--no-captions` rendert ohne Untertitel.

### `clipper run <url>`

Alles am Stück. Zusätzlich zu den `analyze`-Optionen:

| Option | Bedeutung |
|---|---|
| `--no-llm` | Heuristische Auswahl statt Claude |
| `--no-captions` | Ohne eingebrannte Untertitel |
| `--reselect` | Nur die Auswahl neu rechnen, restlichen Cache behalten |

### `clipper list [video-id]`

Zeigt erzeugte Clips mit Score, Hook und fertiger Caption. Ohne Argument alle Videos.

---

## Konfiguration

`config/default.yaml` enthält alle Parameter mit Kommentaren. Eigene Werte per
`--config meine.yaml` — die Datei wird über die Defaults gelegt, du musst nur die
Abweichungen angeben.

### `ingest`

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `max_height` | `1080` | Höhere Auflösung = mehr Crop-Spielraum |
| `download_sections` | `null` | Teilbereich, z. B. `"*120-360"` oder `"2:00-6:00"` |

### `transcribe`

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `model` | `large-v3` | `tiny`…`large-v3`, `distil-large-v3` |
| `device` | `auto` | `auto` \| `cuda` \| `cpu` |
| `compute_type` | `auto` | `float16` auf CUDA, `int8` auf CPU |
| `language` | `null` | `null` = automatische Erkennung |
| `vad_filter` | `true` | Schneidet Stille, verhindert Halluzinationen |

### `scenes`

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `threshold` | `27.0` | Niedriger = mehr erkannte Schnitte |
| `min_scene_len` | `0.4` | Sekunden |

### `select`

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `clips` | `0` | `0` = unbegrenzt, sonst harte Obergrenze |
| `min_score` | `45` | Untergrenze; bei unbegrenzter Zahl die einzige Qualitätsbremse |
| `min_duration` / `max_duration` | `15` / `60` | Sekunden |
| `model` | `claude-opus-5` | Nur für Weg B |
| `effort` | `high` | Nur für Weg B |
| `vision_frames` | `24` | Keyframes fürs Modell, `0` = aus |
| `output_language` | `en` | Sprache von title, hook, caption |

### `reframe`

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `target_width` / `target_height` | `1080` / `1920` | Ausgabeformat |
| `sample_fps` | `4.0` | Analyse-Rate der Motiverkennung |
| `min_face_weight` | `0.001` | Mindestgröße eines Gesichts (Fläche × Score) |
| `static_per_shot` | `true` | Crop bleibt pro Einstellung konstant |

### `captions`

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `enabled` | `true` | |
| `font` / `font_size` | `Arial Black` / `96` | |
| `max_words` | `4` | **Führende Grenze:** Wörter gleichzeitig im Bild |
| `max_chars_per_line` | `20` | Nur noch Überlaufschutz |
| `max_lines` | `2` | |
| `max_gap` | `0.6` | Sprechpause, ab der ein neuer Block beginnt |
| `margin_v` | `420` | Abstand vom unteren Rand |
| `highlight_color` | gelb | Aktives Wort |
| `pop` | `true` | Aktives Wort kurz vergrößern |
| `uppercase` / `strip_punctuation` | `true` / `true` | |
| `hook.*` | | Eigener Block für das Hook-Overlay |

### `render`

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `workers` | `4` | Parallele Renders |
| `encoder` | `auto` | `auto` prüft NVENC **funktional**, sonst libx264 |
| `crf` | `20` | Bei NVENC als `-cq` |
| `fps` | `30` | |
| `loudnorm` | `true` | Auf −14 LUFS, TikTok-Standard |

---

## Design-Entscheidungen

Die interessanten Stellen, jeweils mit der Messung, die dahinter steht.

### Kein PyTorch

Für die Motiverkennung wäre YOLO naheliegend — das zieht rund 3 GB CUDA-Torch nach.
Stattdessen YuNet, das direkt in OpenCV steckt: 340 KB Modell, für Gesichter
gleichwertig. Whisper läuft über CTranslate2, das ebenfalls kein Torch braucht. Die
gesamte Installation bleibt dadurch klein.

### Keyframes gehen als Bilder in die Auswahl

Standardmäßig 24 Stück, ausgewählt zu 60 % an den Audio-Spitzen und zu 40 % über ein
gleichmäßiges Raster. Reine Peak-Auswahl verpasst ruhige Setup-Momente, ein reines
Raster verpasst die Höhepunkte.

### Sequenzielles Dekodieren beim Reframing

Die erste Fassung positionierte den Decoder pro Sample-Frame neu
(`CAP_PROP_POS_MSEC`). Jeder dieser Seeks zwingt zurück auf den letzten Keyframe —
gemessen 0,6 s pro Frame, also **50,2 s Analyse für einen 20-Sekunden-Clip**.

Jetzt wird einmal an den Clipanfang gesprungen, dann sequenziell gegriffen (`grab`),
dekodiert wird nur an den Sample-Punkten (`retrieve`). Gemessen **5,9 s statt 50,2 s,
Faktor 8,5.**

### Crop bleibt pro Shot statisch

Ein Crop, der innerhalb einer Einstellung mitwandert, wirkt bei schnell geschnittenem
Material unruhig und kostet deutlich mehr Rechenzeit. Ein harter Sprung auf der
Schnittgrenze fällt dagegen nicht auf, weil dort ohnehin das ganze Bild wechselt.

Zum Maßstab: ein 20-minütiges MrBeast-Video hat 642 Shots, Median 1,63 s, 17 % davon
unter einer Sekunde.

Die Rückfallkette pro Shot: gewichteter Gesichtsschwerpunkt → Bewegungsschwerpunkt →
Position des vorigen Shots → Bildmitte.

### Mindestgröße für Gesichter

Ohne Schwelle folgte der Crop einmal einer Person weit im Hintergrund, die 0,04 % der
Bildfläche belegte, aber die einzige Erkennung im Shot war — der Ausschnitt knallte an
den Bildrand und verpasste die Szene.

Gemessen über 144 Erkennungen: echte Motive liegen im Median bei 0,54 % der Fläche,
das 10. Perzentil bei 0,15 %. Die Schwelle `min_face_weight: 0.001` behält 95 % der
Erkennungen und filtert genau solche Ausreißer.

### Funktionaler Encoder-Test

Ein einkompilierter Encoder heißt nicht, dass er läuft. NVENC scheitert zur Laufzeit,
wenn der Treiber älter ist als die NVENC-API, gegen die ffmpeg gebaut wurde — das fällt
sonst erst beim Rendern auf. `pick_encoder` kodiert deshalb testweise einen Frame und
fällt bei Fehler auf libx264 zurück.

---

## Maximale Ausbeute pro Video

`select.clips: 0` (Standard) heißt: alles nehmen, was durchkommt. Die Bremse ist keine
Stückzahl, sondern `select.min_score`.

Kollidierende Clips werden nicht einfach fallengelassen. `_finalize` versucht drei
Platzierungen, bevor es aufgibt:

1. Start rückwärts auf die Schnittgrenze davor — beste Qualität, aber kann in den
   Vorgänger ragen
2. Start vorwärts auf die Grenze danach
3. Einpassen in die tatsächlich freie Lücke zwischen zwei bereits vergebenen Clips

**Gemessen an einem 20:44-Minuten-Video:** 27 von 34 Vorschlägen mit nur Schritt 1,
**34 von 34** mit allen dreien. Ergebnis: 14,2 Minuten Clipmaterial, 68 % des
Quellvideos verwertet, Clip-Länge im Median 25,6 s.

`clipper select` nennt verworfene Clips namentlich samt Grund.

---

## Untertitel

Karaoke-Stil: die ganze Zeile steht, das aktive Wort ist farbig und wird kurz
vergrößert. Reine ASS-Tags, kostet keine zusätzliche Rechenzeit.

**`max_words` steuert die Dichte** und ist die führende Grenze. Gemessen an einem Clip:

| Einstellung | Wörter pro Block | Standzeit |
|---|---|---|
| ohne Wortgrenze | 7 | 1,48 s |
| `max_words: 4` (Standard) | 4 | 0,76 s |
| `max_words: 3` | 3 | 0,60 s |

**Umbruch an Sprechpausen:** ab `max_gap` Sekunden Stille beginnt ein neuer Block.
Ohne diese Prüfung bleibt eine Zeile während einer langen Pause tot im Bild stehen.

**Hook-Overlay:** der Hook liegt die ersten Sekunden groß im oberen Bilddrittel, als
eigener ASS-Style statt als `drawtext`-Filter — das erspart das Escaping von
Sonderzeichen im Filtergraph. Zu langer Text wird an der Wortgrenze mit Ellipse
gekürzt, nie mitten im Wort.

> **Achtung bei `max_chars_per_line`:** Der Wert muss zur Schriftgröße passen. Bei
> 96 px in Arial Black passen rund 22 Zeichen in die 1080 px Bildbreite — mehr läuft
> links und rechts aus dem Bild, und ASS bricht wegen `WrapStyle: 2` nicht automatisch
> um. Längere Standzeiten holt man über `max_lines` und `max_words`, nicht über
> breitere Zeilen.

Umlaute und Sonderzeichen funktionieren (`FÜR`, `GRÖSSTE`, `ÄÖÜ`). `ß` wird bei
`uppercase: true` korrekt zu `SS`.

---

## Bekannte Grenzen

**Weite Establisher mit statischem, gesichtslosem Motiv.** Etwa eine Falle am linken
Bildrand vor leerem Sand. Weder Gesichts- noch Bewegungserkennung findet so etwas.

Ein Textur- und Sättigungs-Maximum als Ersatz wurde getestet und wieder verworfen:
gegen 22 Frames mit erkannten Gesichtern geprüft lag es im Median 0,43 Bildbreiten
daneben und traf nur in 14 % der Fälle auf 0,15 genau. Es korreliert schlicht nicht mit
dem Motiv. Ein echter Fix bräuchte ein Objektmodell (opencv-contrib Saliency oder YOLO)
und damit eine deutlich schwerere Installation.

Praktisch ist das meist ein Auswahlproblem, kein Crop-Problem: einen Clip nicht auf
einer Totalen beginnen lassen, sondern auf der Aktion.

**Renderzeit.** 34 Clips (14 Minuten Material) brauchen mit `workers: 4` rund
10 Minuten auf einem i5-13600KF — ohne NVENC, also libx264 auf der CPU. Ein
Treiber-Update, das NVENC freischaltet, ist hier der größte einzelne Hebel.

**Noch offen:** Whisper-Batching (`BatchedInferencePipeline`, Faktor 3–4),
Downscaling vor der Szenenerkennung, Prompt-Caching für `--reselect`,
Längenvarianten desselben Moments als A/B-Test.

---

## Troubleshooting

**`Library cublas64_12.dll is not found`**
Die pip-Pakete legen ihre CUDA-DLLs unter `site-packages/nvidia/*/bin` ab, wo
CTranslate2 sie nicht findet. `utils/cuda.py` registriert die Pfade beim Start —
sowohl per `os.add_dll_directory` als auch über `PATH`, weil CTranslate2 über
`LoadLibraryA` lädt und das nur `PATH` durchsucht. Tritt der Fehler trotzdem auf:
`uv sync --extra cuda` erneut laufen lassen.

**`Driver does not support the required nvenc API version`**
ffmpeg wurde gegen eine neuere NVENC-API gebaut als dein Treiber bereitstellt. Die
Pipeline fällt automatisch auf libx264 zurück. Für GPU-Encoding den NVIDIA-Treiber
aktualisieren.

**`Unable to parse "original_size" option value`**
ffmpeg deutet den Doppelpunkt eines Windows-Laufwerksbuchstabens als Options-Trenner.
Deshalb läuft ffmpeg mit `cwd` = Verzeichnis der ASS-Datei und bekommt nur den
Dateinamen. Sollte nicht mehr auftreten.

**`UnicodeEncodeError` bei `clipper list`**
Die Windows-Konsole läuft auf cp1252 und stolpert über Emojis in Captions. `cli.py`
stellt stdout beim Start auf UTF-8 mit `errors="replace"`.

**Untertitel in falscher Schrift**
libass ersetzt fehlende Fonts stillschweigend. Prüfen, ob `captions.font` wirklich
installiert ist.

**`download_sections` lädt das ganze Video**
Behoben. Die Sternchen-Syntax der yt-dlp-CLI versteht die Python-API nicht, sie wird
jetzt vorher in `(start, end)` in Sekunden aufgelöst.

---

## Tests

```bash
python -m pytest tests/ -q
```

34 Tests. Der Schwerpunkt liegt auf Schnittgrenzen und Untertitel-Layout — dort fällt
ein Fehler erst beim Ansehen des Ergebnisses auf, nicht als Exception. Genau diese
Tests haben während der Entwicklung drei echte Bugs gefunden:

- Blockkapazität rechnete `max_chars × max_lines`, obwohl greedy umgebrochene Zeilen
  am Zeilenende Platz verschenken
- `snap_end_within` konnte im Fallback ein Ende hinter der letzten Shot-Grenze liefern,
  also Clips jenseits des Videoendes
- Die Längenkorrektur zerstörte das vorher berechnete Shot-Snapping

---

## Projektstruktur

```
src/clipper/
  cli.py                 Kommandozeile (analyze, select, build, run, list)
  pipeline.py            Orchestrierung, Caching, Briefing, paralleles Rendern
  config.py              YAML laden und mergen
  models.py              Pydantic-Modelle für alle Zwischenstände
  stages/
    ingest.py            yt-dlp, Abschnitts-Download
    transcribe.py        faster-whisper, CUDA-Auflösung
    scenes.py            PySceneDetect, Snapping-Logik
    audio.py             RMS-Hüllkurve, Peak-Erkennung
    vision.py            Keyframe-Auswahl und -Extraktion
    select.py            Prompt, LLM-Aufruf, Heuristik, Nachbearbeitung
    reframe.py           YuNet, Bewegungsschwerpunkt, Crop-Planung
    captions.py          ASS-Erzeugung, Blocklayout, Hook-Overlay
    render.py            ffmpeg-Filtergraph und Encoding
  utils/
    ffmpeg.py            ffprobe, Encoder-Test, Audio-Dekodierung
    cuda.py              CUDA-DLL-Pfade unter Windows
    cache.py             JSON-Caching der Stufen
config/default.yaml      Alle Parameter mit Kommentaren
tests/test_pipeline.py   34 Tests
work/<video-id>/         Cache: Quelle, Transkript, Shots, Energie, Keyframes
out/<video-id>/          Fertige Clips + clips.json
```

---

## Rechtliches

Die Pipeline lädt fremdes Videomaterial herunter und verarbeitet es. Ob du das Ergebnis
veröffentlichen darfst, hängt allein davon ab, ob dir jemand die Rechte daran eingeräumt
hat — etwa über eine Clipping-Kampagne, die genau dieses Material freigibt.

Ohne eine solche Freigabe gibt es dafür keine Rechtsgrundlage. Ein Fair-Use-Äquivalent
existiert im deutschen Urheberrecht nicht; das Zitatrecht (§ 51 UrhG) ist eng und
verlangt eine eigene inhaltliche Auseinandersetzung, reines Nachschneiden fällt nicht
darunter.

Praktische Konsequenzen:

- Nur das Material verwenden, das der Kampagnen-Brief freigibt — nicht den Katalog
  eines Creators allgemein
- Auflagen einhalten: Wasserzeichen, Tagging, erlaubte Plattformen, Theme-Page-Regeln
- Kampagnen räumen sich in der Regel das Recht ein, jederzeit die Löschung zu verlangen
- Musikrechte sind eine eigene Ebene: ein lizenziertes Video mit nicht lizenziertem
  Sound kann trotzdem gemutet oder gesperrt werden

Dieses Repository stellt ein Werkzeug bereit. Die Verantwortung für die Rechte am
verarbeiteten Material liegt bei dir.

---

## Lizenz

MIT — siehe [LICENSE](LICENSE).
