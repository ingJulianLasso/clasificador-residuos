"""Formato en español de las cifras que van al informe: punto de miles, coma decimal y LaTeX seguro."""

from __future__ import annotations

import re
import unicodedata
from datetime import date

MESES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)  # fmt: skip
DECIMALES_METRICA = 3
MENOS = "$-$"


def entero(numero: int) -> str:
    return f"{int(numero):,}".replace(",", ".")


def decimal(numero: float, cifras: int = 1) -> str:
    texto = f"{abs(numero):,.{cifras}f}".translate(str.maketrans(",.", ".,"))
    return f"{MENOS}{texto}" if numero < 0 and float(texto.replace(".", "").replace(",", ".")) != 0 else texto


def metrica(numero: float) -> str:
    """Métricas en proporción, con tres decimales."""
    return decimal(numero, DECIMALES_METRICA)


def intervalo(par: list[float] | tuple[float, float]) -> str:
    return f"[{metrica(par[0])}; {metrica(par[1])}]"


def con_intervalo(valor: float, par: list[float] | tuple[float, float]) -> str:
    return f"{metrica(valor)} {intervalo(par)}"


def p_valor(p: float) -> str:
    """La expresión completa, como la pide APA: «p = 0,123» con tres decimales, o «p < 0,001»."""
    return "$p < 0{,}001$" if p < 0.001 else f"$p = {decimal(p, 3).replace(',', '{,}')}$"


def fecha(iso: str) -> str:
    dia = date.fromisoformat(iso)
    return f"{dia.day} de {MESES[dia.month - 1]} de {dia.year}"


def identificador(texto: str) -> str:
    """Nombre válido de macro: sin tildes ni espacios, cada palabra con mayúscula inicial."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return "".join(palabra.capitalize() for palabra in re.split(r"[^A-Za-z]+", sin_tildes) if palabra)


def texto_latex(texto: str) -> str:
    """Escapa los caracteres que LaTeX interpreta en un texto corriente."""
    return texto.replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("&", "\\&").replace("_", "\\_").replace("#", "\\#")
