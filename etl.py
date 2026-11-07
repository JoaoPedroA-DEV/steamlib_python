"""ETL — Taxa de alfabetização (15 anos ou mais) por sexo, território e ano (IBGE)."""
import pandas as pd

OBRIGATORIAS = ["Nivel", "Territorio", "Ano", "Sexo", "Taxa_Alfabetizacao"]
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


def carregar_base(arquivo) -> pd.DataFrame:
    """Lê CSV/XLSX, valida colunas, converte tipos e remove duplicatas e valores inválidos.

    Colunas obrigatórias: Nivel (Brasil/Região/UF), Territorio, Ano, Sexo, Taxa_Alfabetizacao.
    Opcionais: UF (sigla), Regiao, Faixa_Etaria (padrão '15+').
    """
    nome = str(getattr(arquivo, "name", arquivo)).lower()
    df = pd.read_excel(arquivo) if nome.endswith(("xlsx", "xls")) else pd.read_csv(arquivo, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    falta = [c for c in OBRIGATORIAS if c not in df.columns]
    if falta:
        raise ValueError(f"Colunas ausentes: {', '.join(falta)}")
    for c in ("UF", "Regiao", "Faixa_Etaria"):
        if c not in df.columns:
            df[c] = "15+" if c == "Faixa_Etaria" else ""
    df["Taxa_Alfabetizacao"] = pd.to_numeric(
        df["Taxa_Alfabetizacao"].astype(str).str.replace("%", "").str.replace(",", "."), errors="coerce")
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce")
    df = df.dropna(subset=["Ano", "Taxa_Alfabetizacao", "Territorio"]).copy()
    df["Ano"] = df["Ano"].astype(int)
    df = df[df["Taxa_Alfabetizacao"].between(0, 100)]
    for c in ("Nivel", "Territorio", "Sexo", "Faixa_Etaria"):
        df[c] = df[c].astype(str).str.strip()
    df["UF"] = df["UF"].fillna("").astype(str).str.strip()
    df["Regiao"] = df["Regiao"].fillna("").astype(str).str.strip()
    return df.drop_duplicates(["Nivel", "Territorio", "Ano", "Sexo", "Faixa_Etaria"]).reset_index(drop=True)


def nivel(df, n, sexo="Total", faixa="15+"):
    return df[(df["Nivel"] == n) & (df["Sexo"] == sexo) & (df["Faixa_Etaria"] == faixa)]


def evolucao_uf(df, sexo="Total") -> pd.DataFrame:
    """Uma linha por UF com colunas por ano e ganho em pontos percentuais (último ano − primeiro)."""
    u = nivel(df, "UF", sexo)
    if u.empty:
        return u
    p = u.pivot_table(index=["Territorio", "UF", "Regiao"], columns="Ano", values="Taxa_Alfabetizacao").reset_index()
    anos = sorted(c for c in p.columns if isinstance(c, (int, float)))
    if len(anos) >= 2:
        p["Ganho_pp"] = p[anos[-1]] - p[anos[0]]
    return p
