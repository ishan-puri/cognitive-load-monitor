import tkinter as tk
from tkinter import messagebox, filedialog
import random
import time
import csv
import math
from collections import deque

# -----------------------------
# Config
# -----------------------------
WINDOW_TITLE = "Cognitive Load Monitor (HCI Demo)"
EVAL_INTERVAL_SEC = 10          # recompute metrics every N seconds
WINDOW_SIZE = 60                # rolling window of seconds for metrics
HIGH_LOAD_THRESHOLD = 70        # 0-100 scale
HIGH_LOAD_SUSTAINED_WINDOWS = 2 # how many consecutive evals before break reminder
IDLE_THRESHOLD_SEC = 3.0        # seconds without key/mouse → considered idle
LOG_FILENAME_DEFAULT = "clm_session_log.csv"

SHORT_PROMPTS = [
    "Type fast and true.",
    "Human computer interaction.",
    "Short prompt for warmup.",
    "Mental models shape use.",
    "Keep your eyes on text."
]

LONG_PROMPTS = [
    "The quick brown fox jumps over the lazy dog to evaluate typing and accuracy.",
    "Cognitive load varies with task difficulty and available working memory resources.",
    "Users rely on feedback, visibility of system status, and clear affordances to succeed.",
    "Consistent interaction patterns reduce error rates and improve learnability over time.",
    "Adaptive interfaces can modify complexity based on performance and stress indicators."
]

# -----------------------------
# Helpers
# -----------------------------
def now():
    return time.perf_counter()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def smoothstep01(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

def normalize_feature(value, lo, hi):
    """Map value in [lo,hi] to 0..1 (with safety)."""
    if hi <= lo:
        return 0.0
    return clamp((value - lo) / (hi - lo), 0.0, 1.0)

# -----------------------------
# App
# -----------------------------
class CognitiveLoadApp:
    def __init__(self, root):
        self.root = root
        root.title(WINDOW_TITLE)
        root.geometry("900x600")

        self.target_text = tk.StringVar()
        self.status_text = tk.StringVar(value="Welcome! Start typing when ready.")
        self.load_text = tk.StringVar(value="Load: --")
        self.kpm_text = tk.StringVar(value="KPM: --")
        self.error_text = tk.StringVar(value="Errors: --")
        self.mouse_text = tk.StringVar(value="MouseSpeed: --  Idle: --s")

        self.current_prompt = ""
        self.start_time = now()

        # Rolling buffers
        self.keystroke_times = deque()        # timestamps of all keystrokes
        self.backspace_times = deque()        # timestamps of backspaces
        self.mouse_samples = deque()          # (t, x, y)
        self.idle_since = now()
        self.last_mouse = None

        self.eval_timer = None
        self.eval_windows_high = 0

        self.session_start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_file = None
        self.log_writer = None

        self._build_ui()
        self._bind_events()

        # First prompt (easier at start)
        self._set_new_prompt(easier=True)

        # Schedule evaluation loop
        self._schedule_eval()

    # ---------- UI ----------
    def _build_ui(self):
        top = tk.Frame(self.root, padx=16, pady=12)
        top.pack(fill="x")

        tk.Label(top, text="Target text:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.lbl_prompt = tk.Label(
            top, textvariable=self.target_text, wraplength=860, justify="left", font=("Segoe UI", 12)
        )
        self.lbl_prompt.pack(fill="x", pady=(2, 10))

        middle = tk.Frame(self.root, padx=16, pady=8)
        middle.pack(fill="x")

        tk.Label(middle, text="Your input:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.txt_input = tk.Text(middle, height=6, font=("Consolas", 12))
        self.txt_input.pack(fill="x")
        self.txt_input.focus_set()

        btns = tk.Frame(self.root, padx=16, pady=8)
        btns.pack(fill="x")
        tk.Button(btns, text="New Prompt", command=self._new_prompt_button).pack(side="left")
        tk.Button(btns, text="Save Log…", command=self._save_log_as).pack(side="left", padx=8)
        tk.Button(btns, text="Clear Input", command=self._clear_input).pack(side="left")

        status = tk.Frame(self.root, padx=16, pady=10)
        status.pack(fill="x")
        tk.Label(status, textvariable=self.status_text, fg="#555").pack(anchor="w")

        metrics = tk.Frame(self.root, padx=16, pady=12)
        metrics.pack(fill="x")
        tk.Label(metrics, textvariable=self.load_text, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        row = tk.Frame(metrics); row.pack(fill="x")
        tk.Label(row, textvariable=self.kpm_text, width=18, anchor="w").pack(side="left")
        tk.Label(row, textvariable=self.error_text, width=22, anchor="w").pack(side="left")
        tk.Label(row, textvariable=self.mouse_text, width=38, anchor="w").pack(side="left")

        footer = tk.Frame(self.root, padx=16, pady=10)
        footer.pack(fill="x")
        tk.Label(
            footer,
            text="Tip: High load triggers shorter prompts and break reminders. Move the mouse and type naturally.",
            fg="#777"
        ).pack(anchor="w")

    def _bind_events(self):
        self.root.bind_all("<Key>", self._on_key_any)
        self.root.bind_all("<BackSpace>", self._on_backspace)
        self.root.bind_all("<Button-1>", self._on_mouse)
        self.root.bind_all("<Motion>", self._on_mouse_motion)

    # ---------- Prompt & Input ----------
    def _set_new_prompt(self, easier=False):
        if easier:
            self.current_prompt = random.choice(SHORT_PROMPTS)
        else:
            # pick longer if performance is good / load low (handled at eval)
            self.current_prompt = random.choice(LONG_PROMPTS)
        self.target_text.set(self.current_prompt)
        self._clear_input()

    def _new_prompt_button(self):
        # manual override: just generate based on last load estimate (if any)
        easier = getattr(self, "last_load_score", 50) >= HIGH_LOAD_THRESHOLD
        self._set_new_prompt(easier=easier)

    def _clear_input(self):
        self.txt_input.delete("1.0", "end")
        self.status_text.set("New prompt loaded. Start typing!")
        self.keystroke_times.clear()
        self.backspace_times.clear()

    # ---------- Event Handlers ----------
    def _on_key_any(self, event):
        t = now()
        self.keystroke_times.append(t)
        self.idle_since = t

        # realtime feedback: color text by correctness
        target = self.current_prompt
        typed = self.txt_input.get("1.0", "end-1c")
        mismatches = self._mismatch_count(typed, target)
        if mismatches == 0:
            self.status_text.set("Good so far ✓")
        else:
            self.status_text.set(f"{mismatches} mismatches")

    def _on_backspace(self, event):
        t = now()
        self.backspace_times.append(t)

    def _on_mouse(self, event):
        t = now()
        self.idle_since = t
        self._sample_mouse(t, event.x_root, event.y_root)

    def _on_mouse_motion(self, event):
        t = now()
        # consider motion as activity only if moved a bit
        self._sample_mouse(t, event.x_root, event.y_root)

    def _sample_mouse(self, t, x, y):
        if self.last_mouse is None:
            self.last_mouse = (t, x, y)
            self.mouse_samples.append((t, x, y))
            return
        lt, lx, ly = self.last_mouse
        # sample at ~50ms min spacing to avoid huge queues
        if (t - lt) >= 0.05:
            self.mouse_samples.append((t, x, y))
            self.last_mouse = (t, x, y)
        # trim window
        while self.mouse_samples and (t - self.mouse_samples[0][0] > WINDOW_SIZE):
            self.mouse_samples.popleft()

    # ---------- Metrics ----------
    def _mismatch_count(self, typed, target):
        # Levenshtein distance (fast restricted) for short prompts: simple DP
        m, n = len(typed), len(target)
        # optimize: if very long, early exit
        if max(m, n) > 300:
            return abs(m - n)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            ti = typed[i - 1]
            for j in range(1, n + 1):
                cost = 0 if ti == target[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,        # deletion
                    curr[j - 1] + 1,    # insertion
                    prev[j - 1] + cost  # substitution
                )
            prev = curr
        return prev[n]

    def _compute_metrics(self):
        t_now = now()

        # Trim keystroke/backspace queues to WINDOW_SIZE
        while self.keystroke_times and (t_now - self.keystroke_times[0] > WINDOW_SIZE):
            self.keystroke_times.popleft()
        while self.backspace_times and (t_now - self.backspace_times[0] > WINDOW_SIZE):
            self.backspace_times.popleft()

        # Keystrokes per minute
        kpm = len(self.keystroke_times) * (60.0 / max(1e-9, min(WINDOW_SIZE, t_now - (self.keystroke_times[0] if self.keystroke_times else t_now))))

        # Error rate: ratio of backspaces to keystrokes (bounded)
        total_keys = max(1, len(self.keystroke_times))
        error_rate = len(self.backspace_times) / total_keys  # 0..1+ (we'll clamp later)

        # Mouse speed: path length / time (pixels per second)
        path = 0.0
        duration = 0.0
        if len(self.mouse_samples) >= 2:
            t0 = self.mouse_samples[0][0]
            t1 = self.mouse_samples[-1][0]
            duration = max(1e-6, t1 - t0)
            px, py = None, None
            for i in range(1, len(self.mouse_samples)):
                _, x0, y0 = self.mouse_samples[i - 1]
                _, x1, y1 = self.mouse_samples[i]
                dx, dy = x1 - x0, y1 - y0
                path += math.hypot(dx, dy)
        mouse_speed = path / duration if duration > 0 else 0.0

        # Idle time
        idle_sec = t_now - self.idle_since

        # Text accuracy vs target
        typed = self.txt_input.get("1.0", "end-1c")
        mismatches = self._mismatch_count(typed, self.current_prompt)
        # normalized accuracy: 1 - edit_distance/len(target)
        denom = max(1, len(self.current_prompt))
        accuracy = 1.0 - clamp(mismatches / denom, 0.0, 1.0)

        # Heuristic cognitive load score 0..100
        # Intuition:
        # - Lower kpm tends to mean higher load (inverted)
        # - Higher error_rate → higher load
        # - Very high or very jittery mouse speed → higher load (U-shaped: too low = idle)
        # - Longer idle → higher load (or disengagement)
        # - Lower accuracy → higher load
        # Normalize features using broad, practical ranges
        kpm_n = 1.0 - normalize_feature(kpm, 120, 360)                # 120..360 KPM typical → invert
        err_n = normalize_feature(error_rate, 0.02, 0.20)             # 2%..20% backspace ratio
        acc_n = 1.0 - accuracy                                        # accuracy deficit
        idle_n = normalize_feature(idle_sec, 1.0, 6.0)                 # >6s idle is heavy
        # Mouse: U-shaped (too still or too jittery)
        ms_low = normalize_feature(mouse_speed, 20, 120)              # 20..120 px/s = "healthy"
        ms_high = normalize_feature(mouse_speed, 300, 1200)           # too jittery
        mouse_u = smoothstep01(1.0 - ms_low) * 0.6 + smoothstep01(ms_high) * 0.4

        # Weighted sum → 0..100
        weights = {
            "kpm": 0.25,
            "err": 0.25,
            "acc": 0.30,
            "idle": 0.10,
            "mouse": 0.10
        }
        load01 = (
            weights["kpm"] * kpm_n +
            weights["err"] * err_n +
            weights["acc"] * acc_n +
            weights["idle"] * idle_n +
            weights["mouse"] * mouse_u
        )
        load_score = int(round(100.0 * clamp(load01, 0.0, 1.0)))

        metrics = {
            "kpm": kpm,
            "error_rate": error_rate,
            "mouse_speed": mouse_speed,
            "idle_sec": idle_sec,
            "accuracy": accuracy,
            "load": load_score,
            "mismatches": mismatches,
            "typed_len": len(typed)
        }
        return metrics

    # ---------- Eval Loop ----------
    def _schedule_eval(self):
        if self.eval_timer is not None:
            self.root.after_cancel(self.eval_timer)
        self.eval_timer = self.root.after(EVAL_INTERVAL_SEC * 1000, self._on_eval_tick)

    def _on_eval_tick(self):
        m = self._compute_metrics()
        self.last_load_score = m["load"]

        self.load_text.set(f"Load: {m['load']} / 100")
        self.kpm_text.set(f"KPM: {m['kpm']:.0f}")
        self.error_text.set(f"Errors: {m['error_rate']*100:.1f}%  Acc: {m['accuracy']*100:.0f}%")
        self.mouse_text.set(f"MouseSpeed: {m['mouse_speed']:.0f}px/s  Idle: {m['idle_sec']:.1f}s")

        self._log_metrics(m)
        self._maybe_adapt(m)
        self._maybe_prompt_break(m)

        self._schedule_eval()

    # ---------- Adaptation & Breaks ----------
    def _maybe_adapt(self, m):
        if m["load"] >= HIGH_LOAD_THRESHOLD:
            # Make task easier: shorter prompt
            self._set_new_prompt(easier=True)
            self.status_text.set("High load detected → providing a shorter prompt.")
        else:
            # If accuracy high and error low, increase difficulty
            if m["accuracy"] >= 0.95 and m["error_rate"] <= 0.05:
                self._set_new_prompt(easier=False)
                self.status_text.set("Great performance → giving a longer prompt.")

    def _maybe_prompt_break(self, m):
        if m["load"] >= HIGH_LOAD_THRESHOLD:
            self.eval_windows_high += 1
        else:
            self.eval_windows_high = 0

        if self.eval_windows_high >= HIGH_LOAD_SUSTAINED_WINDOWS:
            self.eval_windows_high = 0
            messagebox.showinfo(
                "Break Suggestion",
                "Your cognitive load has been high for a while.\nTake a 60–90 second break and hydrate. 😊"
            )

    # ---------- Logging ----------
    def _ensure_log(self):
        if self.log_file is None:
            # open default; user can "Save Log…" later to export a copy
            self.log_file = open(LOG_FILENAME_DEFAULT, "w", newline="", encoding="utf-8")
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow([
                "session_start", "wall_time", "seconds_elapsed",
                "kpm", "error_rate", "accuracy",
                "mouse_speed_px_s", "idle_sec", "load",
                "prompt_len", "typed_len", "mismatches"
            ])

    def _log_metrics(self, m):
        self._ensure_log()
        wall = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = now() - self.start_time
        self.log_writer.writerow([
            self.session_start_wall, wall, f"{elapsed:.1f}",
            f"{m['kpm']:.1f}", f"{m['error_rate']:.3f}", f"{m['accuracy']:.3f}",
            f"{m['mouse_speed']:.1f}", f"{m['idle_sec']:.1f}", f"{m['load']}",
            len(self.current_prompt), m["typed_len"], m["mismatches"]
        ])
        self.log_file.flush()

    def _save_log_as(self):
        if self.log_file is None:
            self._ensure_log()
        path = filedialog.asksaveasfilename(
            title="Save Session Log",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            # copy default log into chosen path
            with open(LOG_FILENAME_DEFAULT, "r", encoding="utf-8") as src, \
                 open(path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            messagebox.showinfo("Saved", f"Log saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save log:\n{e}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = CognitiveLoadApp(root)
    root.mainloop()
