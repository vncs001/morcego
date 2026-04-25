"""Terminal dashboard usando rich — atualiza in-place a cada frame."""

import sys
import io
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box

# força UTF-8 no stdout do Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

COLORS = {
    "drone":      "bold red",
    "bird":       "bold green",
    "person":     "bold yellow",
    "car":        "bold cyan",
    "motorcycle": "bold orange1",
    "unknown":    "dim white",
}

ICONS = {
    "drone":      "DRONE",
    "bird":       "bird ",
    "person":     "pers ",
    "car":        "car  ",
    "motorcycle": "moto ",
    "unknown":    "?    ",
}


def _cv_bar(cv: float, width: int = 20) -> Text:
    filled = min(width, int(cv * width / 0.5))
    bar = "#" * filled + "-" * (width - filled)
    if cv >= 0.25:
        color = "bold red"
    elif cv >= 0.04:
        color = "green"
    else:
        color = "dim white"
    return Text(f"|{bar}| {cv:.3f}", style=color)


def _make_table(frame_results: list, frame: int, elapsed_ms: float,
                correct: int, total: int, active_alerts: list = None) -> Panel:
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        border_style="bright_black",
        expand=True,
    )
    table.add_column("Real",      style="dim white",  width=14)
    table.add_column("Detectado", width=16)
    table.add_column("Conf", justify="right", width=6)
    table.add_column("Vel (m/s)", justify="right", width=10)
    table.add_column("Blade CV", width=32)
    table.add_column("", width=6)

    for row in frame_results:
        true_lbl, cls, feats = row[0], row[1], row[2]
        ok = cls.label == true_lbl
        det_text = Text(f"[{ICONS[cls.label]}] {cls.label}", style=COLORS[cls.label])
        alert = Text("DRONE!", style="bold red") if cls.label == "drone" else Text("")
        status = Text("OK", style="bold green") if ok else Text("MISS", style="bold red")

        table.add_row(
            f"{true_lbl}",
            det_text,
            f"{cls.confidence:.2f}",
            f"{feats.velocity_mps:.1f}",
            _cv_bar(feats.blade_cv),
            alert if cls.label == "drone" else status,
        )

    acc = f"{correct}/{total} ({correct/total*100:.0f}%)" if total else "—"
    title = (
        f"[bold white]Frame [cyan]{frame:04d}[/cyan]"
        f"  [dim]{elapsed_ms:.1f} ms[/dim]"
        f"  Accuracy: [{'green' if correct==total else 'yellow'}]{acc}[/]"
    )

    # linhas de alerta de posição no fim da tabela
    if active_alerts:
        table.add_row("", "", "", "", "", "")
        for pos in active_alerts:
            alert_text = Text(
                f"[CAMERA #{pos.track_id}]  "
                f"{pos.range_m:.0f}m  az={pos.azimuth_deg:+.1f}deg  "
                f"X={pos.x_m:.1f}m Y={pos.y_m:.1f}m  "
                f"V={pos.velocity_mps:.1f}m/s",
                style="bold red",
            )
            table.add_row("", alert_text, "", "", "", "")

    return Panel(table, title=title,
                 border_style="bright_red" if active_alerts else "bright_blue")


class TerminalDisplay:
    def __init__(self, scenario: str, target_names: list[str]):
        self._console = Console(force_terminal=True, highlight=False)
        self._live = Live(console=self._console, refresh_per_second=20, screen=False)
        self._scenario = scenario
        self._targets = target_names

    def __enter__(self):
        header = (
            f"[bold cyan]Drone Radar Detection[/bold cyan]  "
            f"[dim]cenário:[/dim] [yellow]{self._scenario}[/yellow]  "
            f"[dim]alvos:[/dim] [white]{', '.join(self._targets)}[/white]\n"
            "[dim]CV > 0.25 = rotor (drone)  |  CV 0.04–0.25 = asa (pássaro)  |  CV < 0.04 = rígido[/dim]"
        )
        self._console.print(Panel(header, border_style="cyan"))
        self._live.__enter__()
        return self

    def update(self, frame_results: list, frame: int,
               elapsed_ms: float, correct: int, total: int, active_alerts: list = None):
        self._live.update(_make_table(frame_results, frame, elapsed_ms, correct, total,
                                      active_alerts or []))

    def __exit__(self, *args):
        self._live.__exit__(*args)

    def summary(self, latencies: list[float], correct: int, total: int, benchmark: bool):
        import numpy as np
        arr = np.array(latencies)
        self._console.print()
        acc_color = "green" if correct == total else "yellow" if correct / total >= 0.8 else "red"
        self._console.print(
            Panel(
                f"[bold]Accuracy:[/bold] [{acc_color}]{correct}/{total} = {correct/total*100:.1f}%[/]\n"
                + (
                    f"[bold]Latência:[/bold] média [cyan]{arr.mean():.1f}ms[/cyan]  "
                    f"p95 [cyan]{np.percentile(arr,95):.1f}ms[/cyan]  "
                    f"max [cyan]{arr.max():.1f}ms[/cyan]\n"
                    f"[bold]RPi Zero 2 (×5):[/bold] média [yellow]{arr.mean()*5:.1f}ms[/yellow]  "
                    f"→ [yellow]{1000/(arr.mean()*5):.1f} fps[/yellow]"
                    if benchmark else ""
                ),
                title="[bold white]Resultado Final",
                border_style="cyan",
            )
        )
