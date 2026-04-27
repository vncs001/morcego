"""
Sistema de alerta de drone — detecta, registra posição e aciona câmera.

Modos de acionamento da câmera (--camera):
  mock   — imprime o comando no terminal (padrão / simulação)
  http   — faz POST para uma URL (câmera IP ou servidor Flask)
  gpio   — seta pino GPIO HIGH no Raspberry Pi (aciona relé/câmera)
"""

import time
import json
import math
import logging
from dataclasses import dataclass, asdict
from typing import Callable

log = logging.getLogger("alert")


@dataclass
class DronePosition:
    range_m: float          # distância do radar
    azimuth_deg: float      # ângulo horizontal (0=frente, +90=direita)
    velocity_mps: float     # velocidade radial
    confidence: float       # confiança do classificador (0–1)
    track_id: int           # ID da trilha no Kalman tracker
    timestamp: float        # Unix timestamp

    @property
    def x_m(self) -> float:
        """Posição X no plano horizontal (leste)."""
        return self.range_m * math.sin(math.radians(self.azimuth_deg))

    @property
    def y_m(self) -> float:
        """Posição Y no plano horizontal (norte/frente do radar)."""
        return self.range_m * math.cos(math.radians(self.azimuth_deg))

    def to_dict(self) -> dict:
        return {**asdict(self), "x_m": round(self.x_m, 2), "y_m": round(self.y_m, 2)}


# ── Backends de acionamento ───────────────────────────────────────────────────

def _trigger_mock(pos: DronePosition):
    """Simula o acionamento — usado em desenvolvimento."""
    log.info(
        f"[CAMERA] ACIONADA | range={pos.range_m:.0f}m  "
        f"az={pos.azimuth_deg:+.1f}°  "
        f"x={pos.x_m:.1f}m  y={pos.y_m:.1f}m  "
        f"v={pos.velocity_mps:.1f}m/s  conf={pos.confidence:.2f}"
    )


def _trigger_http(url: str) -> Callable:
    """Retorna handler que faz POST para URL da câmera IP."""
    def _trigger(pos: DronePosition):
        try:
            import urllib.request
            payload = json.dumps(pos.to_dict()).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                log.info(f"[CAMERA HTTP] status={resp.status}  pos={pos.to_dict()}")
        except Exception as e:
            log.warning(f"[CAMERA HTTP] falha ao acionar: {e}")
    return _trigger


def _trigger_gpio(pin: int) -> Callable:
    """Seta pino GPIO HIGH por 500ms para acionar câmera/relé (Raspberry Pi)."""
    def _trigger(pos: DronePosition):
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)
            log.info(f"[CAMERA GPIO] pino {pin} HIGH | pos={pos.to_dict()}")
            time.sleep(0.5)
            GPIO.output(pin, GPIO.LOW)
        except ImportError:
            log.warning("[CAMERA GPIO] RPi.GPIO não disponível — use no Raspberry Pi")
        except Exception as e:
            log.warning(f"[CAMERA GPIO] erro: {e}")
    return _trigger


# ── AlertHandler ─────────────────────────────────────────────────────────────

class AlertHandler:
    def __init__(self, mode: str = "mock", camera_url: str = "", gpio_pin: int = 17,
                 cooldown_s: float = 3.0, log_file: str = "detections.log"):
        """
        mode       — "mock", "http" ou "gpio"
        camera_url — URL usada no modo "http"
        gpio_pin   — pino BCM usado no modo "gpio"
        cooldown_s — segundos mínimos entre alertas do mesmo track
        log_file   — arquivo de log das detecções
        """
        self.cooldown_s = cooldown_s
        self._last_alert: dict[int, float] = {}  # track_id → timestamp do último alerta

        # configura backend
        if mode == "http":
            self._trigger = _trigger_http(camera_url)
        elif mode == "gpio":
            self._trigger = _trigger_gpio(gpio_pin)
        else:
            self._trigger = _trigger_mock

        # configura log em arquivo
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding="utf-8"),
            ],
        )

    def check(self, track_id: int, range_m: float, azimuth_deg: float,
              velocity_mps: float, confidence: float) -> DronePosition | None:
        """
        Chama sempre que um drone é classificado.
        Retorna DronePosition se o alerta foi disparado, None se ainda no cooldown.
        """
        now = time.time()
        last = self._last_alert.get(track_id, 0)
        if now - last < self.cooldown_s:
            return None  # cooldown ativo para este track

        pos = DronePosition(
            range_m=round(range_m, 1),
            azimuth_deg=round(azimuth_deg, 1),
            velocity_mps=round(velocity_mps, 1),
            confidence=round(confidence, 3),
            track_id=track_id,
            timestamp=now,
        )
        self._last_alert[track_id] = now
        self._trigger(pos)
        return pos
