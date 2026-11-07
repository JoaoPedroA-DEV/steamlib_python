"""Medidas estatísticas obrigatórias do projeto (NumPy/Pandas)."""
import numpy as np
import pandas as pd


def resumo(valores) -> dict:
    """Média, mediana, moda, quartis, IQR, amplitude, variância, desvio padrão e outliers (1,5 x IQR)."""
    s = pd.Series(valores, dtype="float64").dropna()
    if s.empty:
        raise ValueError("Sem valores para calcular.")
    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    moda = s.mode()
    ddof = 1 if len(s) > 1 else 0
    media = s.mean()
    dp = s.std(ddof=ddof) if len(s) > 1 else 0.0
    return {
        "n": int(len(s)),
        "media": media,
        "mediana": s.median(),
        "moda": moda.iloc[0] if len(moda) and s.value_counts().iloc[0] > 1 else None,
        "q1": q1, "q2": q2, "q3": q3, "iqr": iqr,
        "minimo": s.min(), "maximo": s.max(), "amplitude": s.max() - s.min(),
        "variancia": s.var(ddof=ddof) if len(s) > 1 else 0.0,
        "desvio_padrao": dp,
        "coef_variacao": (dp / media * 100) if media else np.nan,
        "lim_inf": lim_inf, "lim_sup": lim_sup,
    }


def outliers(df: pd.DataFrame, coluna: str, rotulo: str) -> pd.DataFrame:
    """Linhas fora de [Q1 - 1,5 IQR ; Q3 + 1,5 IQR]."""
    r = resumo(df[coluna])
    return df[(df[coluna] < r["lim_inf"]) | (df[coluna] > r["lim_sup"])][[rotulo, coluna]]


def _f(n, d=0):
    return f"{n:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def interpretar(r: dict, dec: int = 0, min_n: int = 4) -> str:
    """Texto interpretativo automático (média x mediana, dispersão, outliers)."""
    if r["n"] < min_n:
        aviso = (f"Atenção: apenas {r['n']} valores — quartis e outliers são pouco confiáveis. "
                 "Carregue uma base com todas as UFs para uma análise robusta. ")
    else:
        aviso = ""
    if r["mediana"]:
        razao = r["media"] / r["mediana"]
        if razao > 1.2:
            assim = (f"A média ({_f(r['media'], dec)}) é {_f(razao,1)}x a mediana ({_f(r['mediana'], dec)}): "
                     "distribuição assimétrica à direita, puxada por valores extremos; "
                     "a mediana é o melhor retrato do valor típico.")
        elif razao < 0.8:
            assim = "A média é menor que a mediana: assimetria à esquerda."
        else:
            assim = "Média e mediana próximas: distribuição relativamente simétrica."
    else:
        assim = ""
    cv = r["coef_variacao"]
    disp = (f" O coeficiente de variação é {_f(cv)}% — "
            f"{'alta heterogeneidade' if cv > 50 else 'heterogeneidade moderada' if cv > 25 else 'baixa heterogeneidade'} "
            "entre os territórios.") if cv == cv else ""
    return aviso + assim + disp


def interpretar_taxa(r: dict, rotulo: str = "UFs", dec: int = 1) -> str:
    """Interpretação para taxas (%), limitadas a 100: usa a distância média–mediana relativa ao IQR."""
    dif = r["mediana"] - r["media"]
    if r["iqr"] and abs(dif) > 0.2 * r["iqr"]:
        forma = ("A média fica abaixo da mediana: poucos valores muito baixos puxam a média para baixo "
                 "(assimetria à esquerda), então a mediana descreve melhor o valor típico.") if dif > 0 else \
                ("A média fica acima da mediana: poucos valores muito altos puxam a média para cima "
                 "(assimetria à direita).")
    else:
        forma = "Média e mediana são próximas: distribuição aproximadamente simétrica."
    return (f"Entre {r['n']} {rotulo}, a mediana é {_f(r['mediana'], dec)}% e a média {_f(r['media'], dec)}%. "
            f"A metade central dos dados fica entre {_f(r['q1'], dec)}% e {_f(r['q3'], dec)}% (IQR de "
            f"{_f(r['iqr'], dec)} p.p.), e a distância entre o melhor e o pior é de {_f(r['amplitude'], dec)} p.p. "
            f"(desvio padrão de {_f(r['desvio_padrao'], dec)} p.p.). {forma}")
