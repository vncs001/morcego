"""Entry point — run the drone-detection radar pipeline."""

import argparse
import time
import numpy as np

from radar_sim import FMCWSimulator, RadarParams
from feature_extractor import extract
from classifier import classify
from tracker import KalmanTracker
from scenarios import CUSTOM_TARGETS, SCENARIOS


def resolve_targets(args):
    """Return list of Target objects from CLI args."""
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

    target_list = resolve_targets(args)

    print(f"\nCenário: {args.scenario or 'custom'}")
    print(f"Alvos  : {[t.label + '@' + str(int(t.range_m)) + 'm' for t in target_list]}")
    print(f"Frames : {args.frames}  SNR: {args.snr} dB\n")

    latencies_ms = []
    correct = 0
    total = 0

    for frame in range(args.frames):
        t0 = time.perf_counter()
        detections = []
        frame_results = []

        for tgt in target_list:
            rdm_db, iq_rc = sim.generate_frame([tgt], snr_db=args.snr)
            feats = extract(rdm_db, iq_rc, params)
            cls = classify(feats)

            det_range = (
                tgt.range_m
                + tgt.velocity_mps * frame * params.chirp_duration * params.num_chirps
            )
            detections.append((det_range, feats.velocity_mps, cls.label))
            frame_results.append((tgt.label, cls, feats))

        tracker.update(detections)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        print(f"--- Frame {frame:03d}  ({elapsed_ms:.1f} ms) ---")
        for true_lbl, cls, feats in frame_results:
            ok = cls.label == true_lbl
            correct += ok
            total += 1
            tag = "OK  " if ok else "MISS"
            alert = " *** DRONE ***" if cls.label == "drone" else ""
            print(
                f"  [{tag}] true={true_lbl:<12s} pred={cls.label:<12s} "
                f"conf={cls.confidence:.2f}  v={feats.velocity_mps:5.1f}m/s  "
                f"cv={feats.blade_cv:.3f}{alert}"
            )

    pct = correct / total * 100 if total else 0
    print(f"\n=== Accuracy: {correct}/{total} = {pct:.1f}% ===")

    if args.benchmark:
        arr = np.array(latencies_ms)
        scale = 5.0
        print("\n=== Benchmark ===")
        print(f"  Mean : {arr.mean():.1f} ms  ({1000/arr.mean():.1f} fps)")
        print(f"  p95  : {np.percentile(arr, 95):.1f} ms")
        print(f"  Max  : {arr.max():.1f} ms")
        print(f"\n  RPi Zero 2 (x{scale:.0f}): {arr.mean()*scale:.1f} ms  "
              f"({1000/(arr.mean()*scale):.1f} fps)")


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
        ),
    )
    parser.add_argument("--scenario", default="all",
                        help="cenário predefinido (ignorado se --targets for usado)")
    parser.add_argument("--targets", nargs="+", metavar="ALVO",
                        help="lista livre de alvos (substitui --scenario)")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--snr", type=float, default=20.0,
                        help="SNR em dB (default 20). Diminua para simular ruído/chuva")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
