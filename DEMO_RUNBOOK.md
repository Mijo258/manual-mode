# Demo-day runbook — Phase 7

PowerShell, `C:\Users\Lenovo\manual-mode`; Python 3.11.9. Close other webcam apps before live calibration or the demo.

```powershell
.\manual_venv\Scripts\python.exe calibrate_demo.py --preflight
.\manual_venv\Scripts\python.exe run_all_tests.py
.\manual_venv\Scripts\python.exe calibrate_demo.py --calibrate
.\manual_venv\Scripts\python.exe av_master.py --mode face
```

Preflight/default tests are headless. Preflight verifies imports, JSON, assets, real models and UDP binding. Read any Phase 1 `WEBCAM SKIPPED` message. Exit calibration before launching the master; selecting a mode automatically opens the separate arrow terminal.

**Architecture:** single `CameraHub` → always-active `SafetyEngine` + exclusive Face/Hand/Eye controller → `GestureEngine` → localhost UDP (`127.0.0.1:5055`) → `arrow_game.py`. Face reuses SafetyEngine landmarks; standby retains DMS.

**Master keys:** `1` Face, `2` Hand, `3` Eye, `c` active-mode neutral, `0` standby, `q` quit. Menu: press Enter after the key; OpenCV windows: keys act immediately, plus `Esc` quits. Hold your head, extended index fingertip or open-eyed pupil at rest and press `c`. Arrow terminal: `r` reset; `q`/`Esc` close game. Quit the master with its own `q`.

| Mode | LEFT | RIGHT | UP | DOWN | STOP |
|---|---|---|---|---|---|
| Face · 478 points | Head left on screen | Head right on screen | Head up | Head down | Neutral around `c` baseline |
| Hand · 21 points | Index fingertip left | Index fingertip right | Index fingertip up | Index fingertip down | Fingertip at calibrated center, fist or curled index |
| Eye · dlib 68 points + shared iris | Pupil left | Pupil right | Pupil up | Pupil down | Pupil at calibrated center or eyes closed |

Hand steering follows the extended index fingertip regardless of other finger counts. Eye steering prefers shared iris X/Y ratios and falls back to standalone dlib pupils; average EAR `<0.19` brakes. Both use EMA smoothing and horizontal dominant-axis tie-breaking around the `[c]` neutral center. Smaller mirrored X/Y means LEFT/UP; larger means RIGHT/DOWN. Debounce: 3 detected frames; tracking loss: STOP after 3 missing frames. Face yaw/pitch are normalized offsets, not degrees.

| Always-on DMS alarm | Default trigger |
|---|---|
| `SLEEP` | EAR `<0.20` continuously for `1.5s` |
| `YAWN` | MAR `>0.50` continuously for `1.5s` |
| `DISTRACTION` | Both iris ratios `<0.25` or both `>0.75` continuously for `2s` |

Cooldown: `2s`; missing/invalid face geometry resets timers.

**10–15 second lighting check:** in `--calibrate`, compare live/rolling EAR and MAR while alert, eyes closed, then yawning. Put thresholds between the corresponding readings. `[1]/[2]` lower/raise EAR `0.01`; `[3]/[4]` lower/raise MAR `0.02`. `[5]/[6]` adjust left gaze and `[7]/[8]` right gaze `0.02`. `[c]` recenters this session; `[r]` resets stats/samples. Optional midpoint samples: `[b]` alert neutral, `[e]` eyes closed, `[y]` yawn, 15 detected frames each. `[s]` saves DMS thresholds **after** `project_config.json.bak` backs up the immediately previous config. Manual EAR/MAR adjustment overrides its sample. `[q]`/`Esc` releases the webcam; rerun preflight after saving.

| Symptom | Quick action |
|---|---|
| False sleep / missed closure | Lower EAR `1` / raise `2` |
| False yawn / missed yawn | Raise MAR with `4` / lower with `3` |
| Camera busy | Close Zoom/Teams/browser camera and the other live project process |
| Model load fails | Check preflight paths: face/hand `.task`, `assets.dlib_shape_predictor_68` |
| Drift / gesture flicker | Recenter `c`; tune `gestures.face`/`gestures.eye` in JSON; restart |
| Audio / safety view unsuitable | Add `--mute` / `--no-safety-window` to master launch; DMS stays active |
| Game unresponsive | Reselect mode; close duplicate arrow terminals; check UDP host/port |
| Undo calibration | Exit live processes; restore `.json.bak` over JSON; rerun preflight |

**Release evidence:** six phases + preflight + helper checks + stress must pass; legacy failures print `LEGACY WARNING`. Stress: 20 NONE → 30 FACE → 30 HAND → 30 EYE → 10 NONE; exactly 120 safety/90 control frames, real UDP, clean stop. Mean/p50/p95/max measure frame processing; switches are separate. Windows peak working set includes runner/models, excludes test subprocesses (`MB = bytes/1024²`). Synthetic black frames measure detector paths; live landmarks/capture/GUI/audio need the live demo.
