"""Adaptador HTTP para el servicio local de Ollama."""
from __future__ import annotations

import requests

URL_OLLAMA = "http://127.0.0.1:11434"
MODELOS = ("llama3.2:3b", "llama3.2:1b")


class ErrorOllama(RuntimeError):
    pass


class ClienteOllama:
    def __init__(self, modelo: str = MODELOS[0]):
        if modelo not in MODELOS:
            raise ValueError("Seleccione uno de los modelos locales documentados.")
        self.modelo = modelo
        self.sesion = requests.Session()
        self.sesion.trust_env = False  # El servicio está en este equipo, sin proxy.

    def modelos(self) -> list[dict]:
        try:
            r = self.sesion.get(f"{URL_OLLAMA}/api/tags", timeout=3)
            r.raise_for_status()
            return r.json()["models"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise ErrorOllama("No se pudo consultar Ollama. Abra Ollama en Windows y vuelva a verificar.") from exc

    def generar(self, sistema: str, mensaje: str) -> str:
        payload = {
            "model": self.modelo, "stream": False, "keep_alive": "1m",
            "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": mensaje}],
            "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 350},
        }
        try:
            r = self.sesion.post(f"{URL_OLLAMA}/api/chat", json=payload, timeout=(3, 180))
            if r.status_code == 404:
                raise ErrorOllama(f"Modelo no disponible. Ejecute: ollama pull {self.modelo}")
            r.raise_for_status()
            contenido = r.json()["message"]["content"]
            if not isinstance(contenido, str) or not contenido.strip():
                raise ErrorOllama("Ollama respondió sin texto; vuelva a intentar.")
            return contenido.strip()
        except requests.Timeout as exc:
            raise ErrorOllama("Ollama superó 180 segundos. Pruebe llama3.2:1b y cierre aplicaciones que consuman RAM.") from exc
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise ErrorOllama("No se obtuvo una respuesta válida de Ollama. Verifique el servicio y el modelo.") from exc
