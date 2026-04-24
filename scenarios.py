"""
Definição de alvos customizados e cenários de teste.

Cada Target tem:
  label          — nome (qualquer string)
  range_m        — distância do radar em metros
  velocity_mps   — velocidade radial em m/s (positivo = se afastando)
  rcs_dbsm       — seção transversal radar (dBsm): carro~+20, pessoa~0, drone~-10, pássaro~-18
  blade_freq_hz  — frequência de rotação das pás (Hz). 0 = sem rotor
  blade_amplitude— intensidade da modulação AM das pás (0.0–1.0)
"""

from radar_sim import Target

# ── Alvos predefinidos ────────────────────────────────────────────────────────

CUSTOM_TARGETS = {

    # ── Drones ──────────────────────────────────────────────────────────────
    "drone_hovering": Target(
        label="drone", range_m=60.0, velocity_mps=0.5, rcs_dbsm=-10.0,
        blade_freq_hz=150.0, blade_amplitude=0.65,
    ),
    "drone_fast": Target(
        label="drone", range_m=100.0, velocity_mps=30.0, rcs_dbsm=-8.0,
        blade_freq_hz=200.0, blade_amplitude=0.65,
    ),
    "drone_small": Target(                          # nano-drone tipo DJI Mini
        label="drone", range_m=40.0, velocity_mps=5.0, rcs_dbsm=-20.0,
        blade_freq_hz=250.0, blade_amplitude=0.55,
    ),
    "drone_large": Target(                          # drone de carga, pás lentas
        label="drone", range_m=120.0, velocity_mps=8.0, rcs_dbsm=0.0,
        blade_freq_hz=80.0, blade_amplitude=0.70,
    ),

    # ── Pássaros ────────────────────────────────────────────────────────────
    "bird_slow": Target(
        label="bird", range_m=30.0, velocity_mps=4.0, rcs_dbsm=-20.0,
        blade_freq_hz=4.0, blade_amplitude=0.10,
    ),
    "bird_fast": Target(
        label="bird", range_m=60.0, velocity_mps=20.0, rcs_dbsm=-15.0,
        blade_freq_hz=10.0, blade_amplitude=0.14,
    ),

    # ── Pessoas ─────────────────────────────────────────────────────────────
    "person_walking": Target(
        label="person", range_m=25.0, velocity_mps=1.5, rcs_dbsm=0.0,
    ),
    "person_running": Target(
        label="person", range_m=40.0, velocity_mps=3.0, rcs_dbsm=1.0,
    ),

    # ── Veículos ────────────────────────────────────────────────────────────
    "car_slow": Target(
        label="car", range_m=80.0, velocity_mps=8.0, rcs_dbsm=18.0,
    ),
    "car_highway": Target(
        label="car", range_m=150.0, velocity_mps=33.0, rcs_dbsm=20.0,
    ),
    "truck": Target(
        label="car", range_m=100.0, velocity_mps=22.0, rcs_dbsm=28.0,
    ),
    "motorcycle_slow": Target(
        label="motorcycle", range_m=70.0, velocity_mps=6.0, rcs_dbsm=6.0,
    ),
}

# ── Cenários prontos ──────────────────────────────────────────────────────────

SCENARIOS = {
    # básicos
    "drone":      ["drone_hovering"],
    "bird":       ["bird_slow"],
    "person":     ["person_walking"],
    "car":        ["car_slow"],
    "motorcycle": ["motorcycle_slow"],

    # cenários compostos
    "all":        ["drone_hovering", "bird_slow", "person_walking", "car_slow", "motorcycle_slow"],

    # cenário urbano: drone sobrevoando trânsito + pedestres
    "urban":      ["drone_hovering", "drone_fast", "person_walking", "person_running",
                   "car_slow", "car_highway", "motorcycle_slow"],

    # pior caso: drone pequeno entre muitos pássaros
    "clutter":    ["drone_small", "bird_slow", "bird_fast", "bird_slow", "bird_fast"],

    # dois drones ao mesmo tempo
    "multi_drone": ["drone_hovering", "drone_fast", "drone_small"],

    # apenas veículos, sem drone (zero falsos positivos esperados)
    "no_drone":   ["car_slow", "car_highway", "truck", "motorcycle_slow", "person_walking"],

    # drone de carga perto de pássaros grandes (RCS similar)
    "hard":       ["drone_large", "bird_fast", "bird_fast", "car_slow"],
}
