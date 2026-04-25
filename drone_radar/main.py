"""Drone radar detection pipeline."""

import argparse
import json
import math
import time
import urllib.request
import numpy as np

from radar_sim import FMCWSimulator, RadarParams
from feature_extractor import extract
from classifier import classify
from tracker import KalmanTracker
from scenarios import CUSTOM_TARGETS, SCENARIOS


def send_alert(camera_url: str, track_id: int, range_m: float,
               azimuth_deg: float, velocity_mps: float, confidence: float):
    """POST coordenadas do drone para o controlador da câmera."""
    x_m = round(range_m * math.sin(math.radians(azimuth_deg)), 2)
    y_m = round(range_m * math.cos(math.radians(azimuth_deg)), 2)
    payload = json.dumps({
        "event":        "drone_detected",
        "track_id":     track_id,
        "range_m":      round(range_m, 1),
        "azimuth_deg":  round(azimuth_deg, 1),
        "x_m":          x_m,
        "y_m":          y_m,
        "velocity_mps": round(velocity_mps, 2),
        "confidence":   round(confidence, 3),
    }).encode()
    try:
        req = urllib.request.Request(
            camera_url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1.0)
        print(f"  [ALERTA ENVIADO] range={range_m:.0f}m az={azimuth_deg:+.1f}° "
              f"x={x_m}m y={y_m}m")
    except Exception as e:
        print(f"  [ALERTA FALHOU] {e}")


def resolve_targets(args):
    if args.targets:
        unknown = [t for t in args.targets if t not in CUSTOM_TARGETS]
        if unknown:
            print(f"[erro] alvos desconhecidos: {unknown}")
            print(f"       disponíveis: {list(CUSTOM_TARGETS.keys())}")
            raise SystemExit(1)
        return [CUSTOM_TARGETS[k] for k in args.targets]
    return [CUSTOM_TARGETS[k] for k in SCENARIOS[args.scenario]]


def run(args):
    params = RadarParams()
    sim = FMCWSimulator(params)
    tracker = KalmanTracker(dt=params.chirp_duration * params.num_chirps)
    target_list = resolve_targets(args)
    last_alert: dict[int, float] = {}   # track_id → timestamp do último alerta

    print(f"\nCenário : {args.scenario or 'custom'}")
    print(f"Alvos   : {[t.label + '@' + str(int(t.range_m)) + 'm' for t in target_list]}")
    print(f"Frames  : {args.frames}  SNR: {args.snr} dB")
    if args.camera_url:
        print(f"Camera  : {args.camera_url}")
    print()

    latencies_ms = []
    correct = 0
    total = 0
    drone_correct = 0
    drone_total = 0

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
            frame_results.append((tgt.label, cls, feats, tgt, det_range))

        tracks = tracker.update(detections)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        print(f"--- Frame {frame:03d}  ({elapsed_ms:.1f} ms) ---")
        frame_has_drone = False
        for true_lbl, cls, feats, tgt, det_range in frame_results:
            correct += cls.label == true_lbl
            total += 1
            if true_lbl == "drone":
                drone_total += 1
                if cls.label == "drone":
                    drone_correct += 1

            if cls.label == "drone":
                frame_has_drone = True
                azimuth = getattr(tgt, "azimuth_deg", 0.0)
                x_m = round(det_range * math.sin(math.radians(azimuth)), 1)
                y_m = round(det_range * math.cos(math.radians(azimuth)), 1)
                print(f"  drone confirmed  "
                      f"range={det_range:.0f}m  az={azimuth:+.1f}°  "
                      f"x={x_m}m  y={y_m}m  "
                      f"v={feats.velocity_mps:.1f}m/s  conf={cls.confidence:.2f}")

                if args.camera_url:
                    track = min(tracks, key=lambda t: abs(t.range_est - det_range), default=None)
                    tid = track.track_id if track else -1
                    now = time.time()
                    if now - last_alert.get(tid, 0) >= args.cooldown:
                        last_alert[tid] = now
                        send_alert(args.camera_url, tid, det_range, azimuth,
                                   feats.velocity_mps, cls.confidence)

        if not frame_has_drone:
            print("  nenhum drone detectado")

    drone_pct = drone_correct / drone_total * 100 if drone_total else 0
    total_pct = correct / total * 100 if total else 0
    print(f"\n--- Taxa de sucesso ---")
    print(f"  Drones : {drone_correct}/{drone_total} = {drone_pct:.0f}%")
    print(f"  Geral  : {correct}/{total} = {total_pct:.0f}%")

    if args.benchmark:
        arr = np.array(latencies_ms)
        print(f"\n--- Benchmark ---")
        print(f"  Mean : {arr.mean():.1f} ms  ({1000/arr.mean():.1f} fps)")
        print(f"  p95  : {np.percentile(arr, 95):.1f} ms")
        print(f"  RPi Zero 2 (x5): {arr.mean()*5:.1f} ms  ({1000/(arr.mean()*5):.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description="Drone radar detection pipeline")
    parser.add_argument("--scenario", default="all", choices=list(SCENARIOS.keys()))
    parser.add_argument("--targets", nargs="+", metavar="ALVO")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--camera-url", default="",
                        help="URL do controlador da câmera — só envia quando drone detectado")
    parser.add_argument("--cooldown", type=float, default=3.0,
                        help="segundos entre alertas do mesmo drone (default: 3)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
