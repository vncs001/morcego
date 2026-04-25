"""
Servidor de API REST para o radar de detecção de drones.

Roda o pipeline de radar em background e notifica o controlador
da câmera via HTTP sempre que um drone for detectado.

Uso:
  python api_server.py
  python api_server.py --scenario clutter --camera-url http://192.168.1.50:8080/alert
  python api_server.py --host 0.0.0.0 --port 5000

Endpoints:
  GET  /status          — estado atual do radar (todos os alvos)
  GET  /stream          — SSE: eventos em tempo real (drone_detected, frame)
  GET  /health          — healthcheck simples
  POST /config          — atualiza camera_url em runtime
"""

import argparse
import json
import math
import queue
import threading
import time
import urllib.request
import logging
from dataclasses import dataclass, asdict

from flask import Flask, jsonify, request, Response

# ── pipeline imports ──────────────────────────────────────────────────────────
from radar_sim import FMCWSimulator, RadarParams
from feature_extractor import extract
from classifier import classify
from tracker import KalmanTracker
from scenarios import CUSTOM_TARGETS, SCENARIOS

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = Flask(__name__)

# ── estado compartilhado (thread-safe via lock) ───────────────────────────────
_lock = threading.Lock()
_state = {
    "running": False,
    "frame": 0,
    "detections": [],       # lista de dicts do frame atual
    "drone_alerts": [],     # últimos alertas de drone (mantém histórico de 20)
    "last_alert_ts": {},    # track_id → timestamp do último push (cooldown)
}
_sse_queue: queue.Queue = queue.Queue(maxsize=100)
_config = {
    "camera_url": "",       # URL do controlador da câmera
    "cooldown_s": 3.0,
    "snr_db": 20.0,
    "scenario": "all",
}


# ── helpers ───────────────────────────────────────────────────────────────────

@dataclass
class DroneAlert:
    track_id: int
    range_m: float
    azimuth_deg: float
    x_m: float
    y_m: float
    velocity_mps: float
    confidence: float
    timestamp: float

    def to_dict(self):
        return asdict(self)


def _push_to_camera(alert: DroneAlert):
    url = _config["camera_url"]
    if not url:
        return
    try:
        payload = json.dumps({"event": "drone_detected", **alert.to_dict()}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            log.info(f"Camera notificada: HTTP {resp.status}")
    except Exception as e:
        log.warning(f"Falha ao notificar câmera: {e}")


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── loop do radar (roda em daemon thread) ─────────────────────────────────────

def _radar_loop(scenario: str, snr_db: float):
    params = RadarParams()
    sim = FMCWSimulator(params)
    tracker = KalmanTracker(dt=params.chirp_duration * params.num_chirps)
    target_list = [CUSTOM_TARGETS[k] for k in SCENARIOS[scenario]]

    with _lock:
        _state["running"] = True

    frame = 0
    while _state["running"]:
        t0 = time.perf_counter()
        detections = []
        frame_detections = []

        for tgt in target_list:
            rdm_db, iq_rc = sim.generate_frame([tgt], snr_db=snr_db)
            feats = extract(rdm_db, iq_rc, params)
            cls = classify(feats)
            det_range = (
                tgt.range_m
                + tgt.velocity_mps * frame * params.chirp_duration * params.num_chirps
            )
            detections.append((det_range, feats.velocity_mps, cls.label))
            frame_detections.append({
                "true_label": tgt.label,
                "detected": cls.label,
                "confidence": round(cls.confidence, 3),
                "range_m": round(det_range, 1),
                "azimuth_deg": getattr(tgt, "azimuth_deg", 0.0),
                "velocity_mps": round(feats.velocity_mps, 2),
                "blade_cv": round(feats.blade_cv, 4),
                "is_drone": cls.label == "drone",
            })

        tracks = tracker.update(detections)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ── verifica drones e dispara alertas ─────────────────────────────
        new_alerts = []
        now = time.time()
        for i, det in enumerate(frame_detections):
            if not det["is_drone"]:
                continue

            # track mais próximo
            track = min(tracks, key=lambda t: abs(t.range_est - det["range_m"]), default=None)
            tid = track.track_id if track else -1

            cooldown = _config["cooldown_s"]
            last = _state["last_alert_ts"].get(tid, 0)
            if now - last < cooldown:
                continue

            az = det["azimuth_deg"]
            r = det["range_m"]
            alert = DroneAlert(
                track_id=tid,
                range_m=r,
                azimuth_deg=az,
                x_m=round(r * math.sin(math.radians(az)), 2),
                y_m=round(r * math.cos(math.radians(az)), 2),
                velocity_mps=det["velocity_mps"],
                confidence=det["confidence"],
                timestamp=now,
            )
            _state["last_alert_ts"][tid] = now
            new_alerts.append(alert)
            log.info(
                f"DRONE DETECTADO | track={tid} range={r:.0f}m "
                f"az={az:+.1f}deg x={alert.x_m}m y={alert.y_m}m "
                f"v={alert.velocity_mps}m/s conf={alert.confidence}"
            )
            # push para câmera em thread separada (não bloqueia o radar)
            threading.Thread(target=_push_to_camera, args=(alert,), daemon=True).start()

        # ── atualiza estado global ─────────────────────────────────────────
        with _lock:
            _state["frame"] = frame
            _state["detections"] = frame_detections
            if new_alerts:
                _state["drone_alerts"] = (
                    [a.to_dict() for a in new_alerts] + _state["drone_alerts"]
                )[:20]

        # ── publica no SSE ─────────────────────────────────────────────────
        frame_payload = {
            "frame": frame,
            "elapsed_ms": elapsed_ms,
            "detections": frame_detections,
        }
        try:
            _sse_queue.put_nowait(("frame", frame_payload))
        except queue.Full:
            pass

        for alert in new_alerts:
            try:
                _sse_queue.put_nowait(("drone_detected", alert.to_dict()))
            except queue.Full:
                pass

        frame += 1
        time.sleep(max(0, 0.05 - (time.perf_counter() - t0)))


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "running": _state["running"]})


@app.get("/status")
def status():
    with _lock:
        return jsonify({
            "frame": _state["frame"],
            "scenario": _config["scenario"],
            "detections": _state["detections"],
            "recent_drone_alerts": _state["drone_alerts"][:5],
        })


@app.get("/stream")
def stream():
    """Server-Sent Events — o cliente recebe eventos em tempo real."""
    def _generate():
        yield _sse_event("connected", {"msg": "stream iniciado"})
        while True:
            try:
                event, data = _sse_queue.get(timeout=30)
                yield _sse_event(event, data)
            except queue.Empty:
                yield ": keepalive\n\n"   # evita timeout do cliente

    return Response(_generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/config")
def config():
    body = request.get_json(silent=True) or {}
    if "camera_url" in body:
        _config["camera_url"] = body["camera_url"]
        log.info(f"camera_url atualizado: {_config['camera_url']}")
    if "cooldown_s" in body:
        _config["cooldown_s"] = float(body["cooldown_s"])
    return jsonify({"ok": True, "config": _config})


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Drone radar API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--scenario", default="all", choices=list(SCENARIOS.keys()))
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--camera-url", default="",
                        help="URL do controlador da câmera (ex: http://192.168.1.50:8080/alert)")
    parser.add_argument("--cooldown", type=float, default=3.0)
    args = parser.parse_args()

    _config["camera_url"] = args.camera_url
    _config["cooldown_s"] = args.cooldown
    _config["scenario"] = args.scenario

    # inicia radar em background
    t = threading.Thread(target=_radar_loop, args=(args.scenario, args.snr), daemon=True)
    t.start()
    log.info(f"Radar iniciado | cenário={args.scenario} SNR={args.snr}dB")
    log.info(f"API em http://{args.host}:{args.port}")
    if args.camera_url:
        log.info(f"Camera controller: {args.camera_url}")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
