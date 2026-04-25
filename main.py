"""Entry point — run the drone-detection radar pipeline."""

import argparse
import time
import numpy as np

from radar_sim import FMCWSimulator, RadarParams
from feature_extractor import extract
from classifier import classify
from tracker import KalmanTracker
from scenarios import CUSTOM_TARGETS, SCENARIOS
from display import TerminalDisplay
from alert import AlertHandler


def resolve_targets(args):
    if args.targets:
        unknown = [t for t in args.targets if t not in CUSTOM_TARGETS]
        if unknown:
            print(f"[erro] alvos desconhecidos: {unknown}")
            print(f"       disponíveis: {list(CUSTOM_TARGETS.keys())}")
            raise SystemExit(1)
        return [CUSTOM_TARGETS[k] for k in args.targets]

    if args.scenario not in SCENARIOS:
        print(f"[erro] cenário desconhecido: {args.scenario}")
        print(f"       disponíveis: {list(SCENARIOS.keys())}")
        raise SystemExit(1)
    return [CUSTOM_TARGETS[k] for k in SCENARIOS[args.scenario]]


def run(args):
    params = RadarParams()
    sim = FMCWSimulator(params)
    tracker = KalmanTracker(dt=params.chirp_duration * params.num_chirps)
    alerter = AlertHandler(
        mode=args.camera,
        camera_url=args.camera_url,
        gpio_pin=args.gpio_pin,
        cooldown_s=args.cooldown,
    )

    target_list = resolve_targets(args)
    scenario_name = args.scenario if not args.targets else "custom"
    target_names = [t.label + "@" + str(int(t.range_m)) + "m" for t in target_list]

    latencies_ms = []
    correct = 0
    total = 0
    active_alerts = []   # posições de drone neste frame

    with TerminalDisplay(scenario_name, target_names) as display:
        for frame in range(args.frames):
            t0 = time.perf_counter()
            detections = []
            frame_results = []
            active_alerts = []

            for tgt in target_list:
                rdm_db, iq_rc = sim.generate_frame([tgt], snr_db=args.snr)
                feats = extract(rdm_db, iq_rc, params)
                cls = classify(feats)

                det_range = (
                    tgt.range_m
                    + tgt.velocity_mps * frame * params.chirp_duration * params.num_chirps
                )
                detections.append((det_range, feats.velocity_mps, cls.label))
                frame_results.append((tgt.label, cls, feats, tgt))

            tracks = tracker.update(detections)

            # ── aciona câmera para cada drone detectado ───────────────────
            for true_lbl, cls, feats, tgt in frame_results:
                correct += cls.label == true_lbl
                total += 1

                if cls.label == "drone":
                    # encontra o track correspondente pela proximidade de range
                    det_range = (
                        tgt.range_m
                        + tgt.velocity_mps * frame * params.chirp_duration * params.num_chirps
                    )
                    track = min(tracks, key=lambda t: abs(t.range_est - det_range), default=None)
                    track_id = track.track_id if track else -1

                    pos = alerter.check(
                        track_id=track_id,
                        range_m=det_range,
                        azimuth_deg=tgt.azimuth_deg,
                        velocity_mps=feats.velocity_mps,
                        confidence=cls.confidence,
                    )
                    if pos:
                        active_alerts.append(pos)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)

            display.update(frame_results, frame, elapsed_ms, correct, total, active_alerts)
            time.sleep(0.05)

    display.summary(latencies_ms, correct, total, args.benchmark)


def main():
    parser = argparse.ArgumentParser(
        description="Drone radar detection pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Cenários prontos (--scenario):\n"
            + "\n".join("  " + k for k in SCENARIOS)
            + "\n\nAlvos disponíveis (--targets):\n"
            + "\n".join("  " + k for k in CUSTOM_TARGETS)
            + "\n\nExemplos:\n"
            "  python main.py --scenario urban\n"
            "  python main.py --scenario clutter --frames 30\n"
            "  python main.py --targets drone_hovering car_highway bird_fast\n"
            "  python main.py --targets drone_small bird_fast bird_fast --snr 10\n"
            "  python main.py --scenario all --camera http --camera-url http://192.168.1.50:5000/trigger\n"
            "  python main.py --scenario all --camera gpio --gpio-pin 17\n"
        ),
    )
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--targets", nargs="+", metavar="ALVO")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--camera", choices=["mock", "http", "gpio"], default="mock",
                        help="modo de acionamento da câmera (default: mock)")
    parser.add_argument("--camera-url", default="http://localhost:5000/trigger",
                        help="URL para modo --camera http")
    parser.add_argument("--gpio-pin", type=int, default=17,
                        help="pino BCM para modo --camera gpio (Raspberry Pi)")
    parser.add_argument("--cooldown", type=float, default=3.0,
                        help="segundos entre alertas do mesmo drone (default: 3)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
