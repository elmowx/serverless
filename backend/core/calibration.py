from __future__ import annotations

ColdStartPhases = dict[str, tuple[float, float]]

HUAWEI_P5_P95: ColdStartPhases = {
    "env_init": (0.117072, 27.5678056),
    "code_loading": (1.0, 2369.0),
    "runtime_start": (47.8174772, 1374.0705692),
    "function_init": (0.0, 1.0),
}


def default_phases() -> ColdStartPhases:
    return dict(HUAWEI_P5_P95)
