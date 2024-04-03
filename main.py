import os
import calendar
import numpy as np
import pandas as pd
import dask.dataframe as dd
import numpy_financial as npf
from datetime import datetime
from scipy.optimize import minimize
from dask.distributed import Client
from pandas.tseries.offsets import MonthEnd
from statsmodels.tsa.seasonal import seasonal_decompose

"""
OBJETIVO: Projeção de Mini e Microgeração Distribuída
O resultado do modelo 4MD são projeções de capacidade instalada, número de adotantes e geração de energia mensal
"""

''' Bibliotecas '''


def get_fluxo_de_caixa(row, vida_util, disponibilidade_kwh_mes, inflacao, ano_troca_inversor,
                       desconto_capex_local, anos_desconto, pagamento_disponibilidade, taxa_desconto_nominal,
                       fator_construcao, tarifas, premissas_reg):
    """Roda um fluxo de caixa para cada caso e retorna métricas financeiras.

        Args:
            row (dataframe): Uma linha do dataframe que contém a base gerada pela função [get_casos_payback]
            vida_util (int): Vida útil do sistema fotovoltaico em anos.
            disponibilidade_kwh_mes (float): Consumo de disponbilidade do consumidor em kWh/mês. Default igual a 100,
                                            equivalente a um consumidor trifásico. Tem efeito somente até o ano de 2022.
            inflacao (float): Taxa anual de inflacao considerada no reajuste das tarifas e para calcular o retorno real
                                  de projetos.
            ano_troca_inversor (int): Ano, a partir do ano de instalação, em que é realizada a troca do inversor
                                          fotovoltaico.
            desconto_capex_local (float): Percentual de desconto a ser aplicado no CAPEX de sistemas de geração
                                              local (ex: 0.1) para simulação de incentivos.
            anos_desconto (list): Anos em que há a incidência do desconto no CAPEX. Ex: c(2024, 2025). Caso não se
                                  aplique, informar 0.
            pagamento_disponibilidade (float): Percentual de meses em que o consumidor residencial paga custo de
                                               disponbilidade em função da variabilidade da geração fotovoltaica.
                                               Tem efeito somente até o ano de 2022.
            taxa_desconto_nominal (float): Taxa de desconto nominal considerada nos cálculos de payback descontado.
                                           Default igual a 0.13.
            fator_construcao (dataframe): Input de fator de Construção do investimento.
            tarifas (dataframe): Input de tarifas.
            premissas_reg (dataframe): Input de premissas regulatórias para serem consideradas nos cálculos.

        Returns:
            pd.DataFrame: Métricas financeiras para cada caso.
    """

    # Dataframe para o fluxo de caixa
    fluxo_caixa = pd.DataFrame({
        "ano_simulacao": range(1, vida_util + 1),
        "segmento": row['segmento'],
        "disco": row['disco'],
        "ano": row['ano']
    })

    # Acrescentando o fator construção no fluxo de caixa
    fluxo_caixa = pd.merge(fluxo_caixa, fator_construcao, on='segmento', how="left")
    fluxo_caixa['fator_construcao'] = np.where(fluxo_caixa["ano_simulacao"] == 1, fluxo_caixa["fator_construcao"], 1)

    # Unificando fluxo de caixa com premissa regulatória, com tarifas e com casos payback
    fluxo_caixa = fluxo_caixa.merge(premissas_reg, on='ano', how="left")
    fluxo_caixa = fluxo_caixa.merge(tarifas, on=['ano', 'disco', 'segmento', 'alternativa'], how="left")

    # Formatando colunas de interesse
    fluxo_caixa['tarifa_autoc_tusd'] = np.where(fluxo_caixa["binomia"],
                                                fluxo_caixa["tarifa_autoc_bin_tusd"],
                                                fluxo_caixa["tarifa_autoc_tusd"]
                                                )
    fluxo_caixa['tarifa_demanda'] = np.where(fluxo_caixa["demanda_g"],
                                             fluxo_caixa["tarifa_demanda_g"],
                                             fluxo_caixa["tarifa_demanda_c"])
    fluxo_caixa["consumo_disponibilidade"] = \
        np.where((fluxo_caixa["segmento"].isin(["comercial_at", "comercial_at_remoto"])) | (fluxo_caixa["binomia"]),
                 0,
                 disponibilidade_kwh_mes
                 )
    fluxo_caixa.drop(['tarifa_demanda_c', 'tarifa_demanda_g', 'tarifa_autoc_bin_tusd'], axis=1, inplace=True)

    # Incluindo inflação no fluxo de caixa
    fluxo_caixa["energia_mes"] = \
        row['geracao_kwh_ano'] * fluxo_caixa['fator_construcao'] * (1 + row['degradacao']) ** (
                1 - fluxo_caixa['ano_simulacao'])

    fluxo_caixa["energia_autoc"] = fluxo_caixa['energia_mes'] * row['fator_autoconsumo']

    fluxo_caixa['energia_inj'] = fluxo_caixa['energia_mes'] - fluxo_caixa['energia_autoc']

    # Ajuste da taxa de inflação ao longo do tempo
    fluxo_caixa['taxa_inflacao'] = (1 + inflacao) ** (fluxo_caixa['ano_simulacao'] - 1)

    # Ajustando valor do capex inicial
    fluxo_caixa['capex'] = 0
    fluxo_caixa['troca_inversor'] = 0
    fluxo_caixa['capex'] = np.where(fluxo_caixa['ano_simulacao'] == 1, -row['capex_inicial'] * 0.8, 0)
    fluxo_caixa['troca_inversor'] = np.where(fluxo_caixa['ano_simulacao'] == ano_troca_inversor,
                                             -row['capex_inversor'], 0)

    # Aplicar desconto no capex
    fluxo_caixa['desconto_capex'] = [desconto_capex_local] * len(fluxo_caixa)

    fluxo_caixa['capex'] = \
        np.where(((row['segmento'] in ["residencial", "comercial_bt"]) | (row['ano'] in anos_desconto)),
                 fluxo_caixa['capex'] * (1 - fluxo_caixa['desconto_capex']),
                 fluxo_caixa['capex']
                 )

    # Calculo do fluxo de caixa
    fluxo_caixa['receita_autoc'] = \
        (fluxo_caixa['taxa_inflacao'] * fluxo_caixa['energia_autoc'] *
         (fluxo_caixa['tarifa_autoc_tusd'] + fluxo_caixa['tarifa_autoc_te'])) / (1 - fluxo_caixa['impostos_cheio'])

    fluxo_caixa['receita_inj_completa'] = \
        fluxo_caixa['taxa_inflacao'] * fluxo_caixa['energia_inj'] * fluxo_caixa['tarifa_inj_te'] / \
        (1 - fluxo_caixa['impostos_te']) + \
        (fluxo_caixa['taxa_inflacao'] * fluxo_caixa['energia_inj'] * fluxo_caixa['tarifa_inj_tusd'] /
         (1 - fluxo_caixa['impostos_tusd']))

    fluxo_caixa['pag_compensacao'] = \
        ((fluxo_caixa['taxa_inflacao'] * fluxo_caixa['energia_inj'] * fluxo_caixa['pag_inj_te'] /
          (1 - fluxo_caixa['impostos_te'])) +
         (fluxo_caixa['taxa_inflacao'] * fluxo_caixa['energia_inj'] * fluxo_caixa['pag_inj_tusd'] /
          (1 - fluxo_caixa['impostos_tusd']))) * -fluxo_caixa['p_transicao']

    fluxo_caixa['demanda_contratada'] = \
        -fluxo_caixa['taxa_inflacao'] * row['pot_sistemas_kw'] * fluxo_caixa['tarifa_demanda'] * 12 / \
        (1 - fluxo_caixa['impostos_cheio'])

    fluxo_caixa['custo_disponibilidade'] = \
        np.where(fluxo_caixa['ano'] <= 2022, -pagamento_disponibilidade * 12 * fluxo_caixa['consumo_disponibilidade'] *
                 fluxo_caixa['fator_construcao'] * fluxo_caixa['taxa_inflacao'] *
                 (fluxo_caixa['tarifa_autoc_tusd'] + fluxo_caixa['tarifa_autoc_te']) / (1 -
                                                                                        fluxo_caixa['impostos_cheio']),
                 0
                 )

    # Incluindo O&M
    fluxo_caixa['oem'] = -row['oem_anual'] * row['capex_inicial'] * fluxo_caixa['fator_construcao']

    fluxo_caixa['saldo_anual'] = \
        fluxo_caixa['capex'] + fluxo_caixa['receita_autoc'] + fluxo_caixa['receita_inj_completa'] + \
        fluxo_caixa['pag_compensacao'] + fluxo_caixa['demanda_contratada'] + fluxo_caixa['oem'] + \
        fluxo_caixa['troca_inversor'] + fluxo_caixa['custo_disponibilidade']

    fluxo_caixa['saldo_acumulado'] = list(np.cumsum(fluxo_caixa['saldo_anual'].tolist()))

    fluxo_caixa['taxa_desconto'] = (1 + taxa_desconto_nominal) ** (fluxo_caixa['ano_simulacao'] - 1)

    fluxo_caixa['saldo_anual_desc'] = fluxo_caixa['saldo_anual'] / fluxo_caixa['taxa_desconto']
    fluxo_caixa['saldo_acumulado_desc'] = list(np.cumsum(fluxo_caixa['saldo_anual_desc'].tolist()))

    # Calculando as métricas separadamente
    temp_payback_inteiro = (fluxo_caixa['saldo_acumulado'] < 0).sum()
    temp_menor_positivo = fluxo_caixa.loc[fluxo_caixa['saldo_acumulado'] > 0, 'saldo_acumulado'].min()
    temp_maior_negativo = fluxo_caixa.loc[fluxo_caixa['saldo_acumulado'] < 0, 'saldo_acumulado'].max()
    temp_payback_frac = -temp_maior_negativo / (temp_menor_positivo - temp_maior_negativo)
    payback = temp_payback_inteiro + temp_payback_frac

    temp_payback_inteiro_desc = (fluxo_caixa['saldo_acumulado_desc'] < 0).sum()
    temp_menor_positivo_desc = fluxo_caixa.loc[fluxo_caixa['saldo_acumulado_desc'] > 0, 'saldo_acumulado_desc'].min()
    temp_maior_negativo_desc = fluxo_caixa.loc[fluxo_caixa['saldo_acumulado_desc'] < 0, 'saldo_acumulado_desc'].max()
    temp_payback_frac_desc = -temp_maior_negativo_desc / (temp_menor_positivo_desc - temp_maior_negativo_desc)
    payback_desc = temp_payback_inteiro_desc + temp_payback_frac_desc

    # Criando um DataFrame com as métricas calculadas
    metricas = pd.DataFrame({
        'disco': row['disco'],
        'segmento': row['segmento'],
        'ano': row['ano'],
        'temp_payback_inteiro': [temp_payback_inteiro],
        'temp_menor_positivo': [temp_menor_positivo],
        'temp_maior_negativo': [temp_maior_negativo],
        'temp_payback_frac': [temp_payback_frac],
        'payback': [payback],
        'temp_payback_inteiro_desc': [temp_payback_inteiro_desc],
        'temp_menor_positivo_desc': [temp_menor_positivo_desc],
        'temp_maior_negativo_desc': [temp_maior_negativo_desc],
        'temp_payback_frac_desc': [temp_payback_frac_desc],
        'payback_desc': [payback_desc]
    })
    metricas = metricas[['disco', 'segmento', 'ano', 'payback', 'payback_desc']]

    # Calculando a TIR nominal
    metricas['tir_nominal'] = npf.irr(fluxo_caixa['saldo_anual'])
    metricas['tir_nominal'] = np.where(np.isnan(metricas['tir_nominal']) & (metricas['payback'] == 25),
                                       -0.2,
                                       metricas['tir_nominal'])

    # Calculando a TIR real
    metricas['tir_real'] = (1 + metricas['tir_nominal']) / (1 + inflacao) - 1

    return metricas


def get_tarifas(tarifas, ano_max_resultado):
    """Roda um fluxo de caixa para cada caso e retorna métricas financeiras.

        Args:
            tarifas (dataframe): Base com as tarifas de demanda e consumo.
            ano_max_resultado (int): Ano final para apresentação dos resultados. Máximo igual a 2060. Default igual a
                                     2060.

        Returns:
            pd.DataFrame: Tarifas de demanda e consumo.
    """

    # Tarifa comercial AT
    tarifas_comercial_at = tarifas[tarifas['subgrupo'] == "A4"].copy()
    tarifas_comercial_at['tarifa_demanda_g'] = [0] * len(tarifas_comercial_at)
    tarifas_comercial_at['tarifa_demanda_c'] = [0] * len(tarifas_comercial_at)
    tarifas_comercial_at['segmento'] = ["comercial_at"] * len(tarifas_comercial_at)

    # Tarifa comercial demanda A
    tarifas_demanda_a = tarifas[tarifas['subgrupo'] == "A4"].copy()
    tarifas_demanda_a = tarifas_demanda_a[['disco', 'ano', 'tarifa_demanda_c', 'tarifa_demanda_g']]
    tarifas_demanda_a.drop_duplicates(inplace=True)

    # Tarifa Comercial AT remoto
    tarifas_comercial_at_remoto = tarifas[tarifas['subgrupo'] == "B3"].copy()
    tarifas_comercial_at_remoto['segmento'] = ["comercial_at_remoto"] * len(tarifas_comercial_at_remoto)
    tarifas_comercial_at_remoto.drop(['tarifa_demanda_c', 'tarifa_demanda_g'], axis=1, inplace=True)
    tarifas_comercial_at_remoto = pd.merge(tarifas_comercial_at_remoto, tarifas_demanda_a, on=['disco', 'ano'])

    # Tarifa Residencial
    tarifas_residencial = tarifas[tarifas['subgrupo'] == "B1"].copy()
    tarifas_residencial['segmento'] = ["residencial"] * len(tarifas_residencial)

    # Tarifa residencial remoto
    tarifas_residencial_remoto = tarifas[tarifas['subgrupo'] == "B1"].copy()
    tarifas_residencial_remoto['segmento'] = ["residencial_remoto"] * len(tarifas_residencial_remoto)

    # Tarifa comercial BT
    tarifas_comercial_bt = tarifas[tarifas['subgrupo'] == "B3"].copy()
    tarifas_comercial_bt['segmento'] = ["comercial_bt"] * len(tarifas_comercial_bt)

    # Fazendo um único dataframe com as tarifas
    tarifas = pd.concat(
        [tarifas_comercial_at, tarifas_comercial_at_remoto, tarifas_residencial, tarifas_residencial_remoto,
         tarifas_comercial_bt], ignore_index=True)

    # Filtrando os anos de interesse
    tarifas = tarifas[tarifas['ano'] <= ano_max_resultado]

    return tarifas


def get_proj_potencia(lista_adotantes, dir_dados_premissas):
    """Realiza a projecao da capacidade instalada de micro e minigeracao distribuida

        Args:
            lista_adotantes (list): Resultado da função [get_proj_adotantes].
            dir_dados_premissas (string): Diretório onde se encontram as premissas.

        Returns:
            dataframe: "proj_potencia" possui os resultados da projeção de capacidade instalada de micro e minigeração
            distribuída.
    """

    # Extrair proj_potencia da lista
    results_proj_adotantes = lista_adotantes[0]

    # Ler dados_gd_historico
    dados_gd = pd.read_excel(os.path.join(dir_dados_premissas, "base_mmgd.xlsx"), header=0, sheet_name='Sheet1')
    potencia_tipica = pd.read_excel(os.path.join(dir_dados_premissas, "potencia_tipica.xlsx"),
                                    header=0, sheet_name='Sheet1')

    # Calcular potência média
    potencia_media = (
        dados_gd.groupby(['disco', 'segmento', 'fonte_resumo'])
        .agg(pot_total=('potencia_instalada_k_w', 'sum'),
             adotantes_total=('num_geradores', 'sum'))
        .assign(pot_media=lambda x: x['pot_total'] / x['adotantes_total'])
    ).reset_index()
    potencia_media = (
        potencia_media
        .pivot_table(index=['disco', 'segmento', 'fonte_resumo'], fill_value=None)
        .assign(pot_media=lambda x: x['pot_media'].fillna(x['pot_media'].mean()))
    ).reset_index()
    potencia_media = potencia_media.loc[:, ['disco', 'segmento', 'fonte_resumo', 'pot_media']]
    potencia_media = potencia_media.merge(potencia_tipica, on=['disco', 'segmento'], how='left')
    potencia_media['pot_media'] = \
        potencia_media.apply(lambda x: x['pot_sistemas_kw'] if pd.isna(x['pot_media']) else x['pot_media'], axis=1)
    potencia_media = potencia_media.drop(columns='pot_sistemas_kw')

    # Juntar com a projeção de adotantes
    proj_potencia = pd.merge(results_proj_adotantes, potencia_media,
                             on=['disco', 'segmento', 'fonte_resumo'], how='left')

    # Histórico de adotantes para substituir anos iniciais da projeção
    dados_gd['date'] = dados_gd.apply(lambda lin: datetime(lin['ano'], lin['mes'], 1), axis=1)
    historico_pot_fontes = (
        dados_gd
        .groupby(['date', 'ano', 'mes', 'disco', 'segmento', 'fonte_resumo'])
        .agg(pot_hist=('potencia_instalada_k_w', 'sum'))
        .reset_index()
        .pivot_table(index=['ano', 'mes', 'disco', 'segmento', 'fonte_resumo'], fill_value=0)
        .reset_index()
    )

    proj_potencia['pot_mes'] = proj_potencia['adotantes_mes'] * proj_potencia['pot_media']

    proj_potencia = pd.merge(proj_potencia, historico_pot_fontes,
                             on=["disco", "segmento", "ano", "mes", "fonte_resumo"], how="left")

    proj_potencia['pot_mes'] = np.where(proj_potencia['date'] <= datetime(2024, 2, 1),
                                        proj_potencia['pot_hist'],
                                        proj_potencia['pot_mes'])

    proj_potencia['pot_mes_mw'] = proj_potencia['pot_mes'] / 1000

    proj_potencia = (
        proj_potencia
        .groupby(['disco', 'segmento', 'fonte_resumo'])
        .apply(lambda x: x.assign(pot_acum_mw=x['pot_mes_mw'].cumsum()))
        .reset_index(drop=True)
    )

    return proj_potencia


def get_proj_adotantes(results_casos_otm, input_consumidores_totais, input_consumidores_nicho, dir_dados_premissas):
    """Realiza a projeção do número de adotantes de micro e minigeração distribuída

        Args:
            results_casos_otm (dataframe): Resultado da função [get_calibra_curva_s].
            input_consumidores_totais (dataframe): Resultado da função [get_mercado_potencial].
            input_consumidores_nicho (dataframe): Resultado da função [get_mercado_potencial].
            dir_dados_premissas (string): Diretório onde se encontram as premissas.

        Returns:
            list: lista com dois dataframes. "proj_adotantes" possui os resultados da projeção de adotantes de micro e
                  minigeração distribuída. "part_adotantes" possui o resultado em termos de participação do número de
                  adotantes frente ao total de unidades consumidoras.
    """

    # Ler dados_gd_historico
    dados_gd = pd.read_excel(os.path.join(dir_dados_premissas, "base_mmgd.xlsx"), header=0, sheet_name='Sheet1')

    # Calcula adotantes_acum e adotantes_mes
    projecao = results_casos_otm.groupby(['disco', 'segmento']).apply(
        lambda x: x.assign(
            adotantes_mes=np.where(
                x['ano'] == 2013,
                x['mercado_potencial'] * x['Ft'],
                (x['mercado_potencial'] * x['Ft']) - (x['mercado_potencial'] * x['Ft']).shift()
            ),
            adotantes_acum=(x['mercado_potencial'] * x['Ft']).cumsum()
        )
    ).reset_index(drop=True)
    projecao['adotantes_mes'] = np.where(projecao['adotantes_mes'] < 2013, 0, projecao['adotantes_mes'])
    projecao['adotantes_mes'] = projecao['adotantes_mes'].astype(int)
    projecao['adotantes_acum'] = projecao['adotantes_acum'].astype(int)

    # Suavização em caso de adotantes = 0
    projecao['adotantes_mes_media'] = \
        projecao.groupby(['disco', 'segmento'])['adotantes_mes'].rolling(window=2,
                                                                         min_periods=1).mean().reset_index(drop=True)
    projecao['adotantes_mes_media'] = \
        np.where(projecao['adotantes_mes_media'].isna(), projecao['adotantes_mes'], projecao['adotantes_mes_media'])
    projecao['adotantes_mes_media'] = projecao['adotantes_mes_media'].astype(int)

    projecao['adotantes_mes_c'] = \
        np.where((projecao['adotantes_mes'] == 0) & (projecao['ano'] > 2019),
                 projecao['adotantes_mes_media'],
                 np.where((projecao['adotantes_mes'].shift() == 0) & (projecao['ano'] > 2019),
                          projecao['adotantes_mes_media'],
                          projecao['adotantes_mes']))
    projecao['adotantes_acum_c'] = projecao.groupby(['disco', 'segmento'])['adotantes_mes_c'].cumsum()
    projecao.drop(['adotantes_mes', 'adotantes_acum', 'adotantes_mes_media'], axis=1, inplace=True)
    projecao.rename(columns={'adotantes_mes_c': 'adotantes_mes', 'adotantes_acum_c': 'adotantes_acum'}, inplace=True)

    # Abertura dos adotantes por fonte
    part_adot_fontes = (
        dados_gd.groupby(['disco', 'segmento', 'fonte_resumo'])
        .agg(adotantes_hist=('num_geradores', 'sum'))
        .reset_index()
        .assign(
            adotantes_hist_total=lambda x: x.groupby(['disco', 'segmento'])['adotantes_hist'].transform('sum'),
            part_fonte=lambda x: x['adotantes_hist'] / x['adotantes_hist_total']
        )
    )

    # Completar valores ausentes
    part_adot_fontes = part_adot_fontes.groupby(['disco', 'segmento', 'fonte_resumo']).apply(
        lambda x: x.assign(
            faltantes=x['part_fonte'].isna().sum()
        )
    ).reset_index(drop=True)

    part_adot_fontes['part_fonte'] = \
        np.where((part_adot_fontes['faltantes'] == 4) & (part_adot_fontes['fonte_resumo'] == 'Fotovoltaica'),
                 1,
                 part_adot_fontes['part_fonte'])

    part_adot_fontes['part_fonte'] = part_adot_fontes['part_fonte'].fillna(0)

    part_adot_fontes = part_adot_fontes[['disco', 'segmento', 'fonte_resumo', 'part_fonte']]

    # Histórico de adotantes para substituir anos iniciais da projeção
    dados_gd['date'] = dados_gd.apply(lambda lin: datetime(lin['ano'], lin['mes'], 1), axis=1)

    historico_adot_fontes = (
        dados_gd
        .groupby(['date', 'ano', 'mes', 'disco', 'segmento', 'fonte_resumo'])
        .agg(adotantes_hist=('num_geradores', 'sum'))
        .reset_index()
    )

    historico_adot_fontes = (
        historico_adot_fontes
        .pivot_table(index=['date', 'ano', 'mes', 'disco', 'segmento', 'fonte_resumo'], fill_value=0)
        .reset_index()
    )

    # Juntar com projecao e historico_adot_fontes
    projecao = (
        projecao
        .merge(part_adot_fontes, on=['disco', 'segmento', 'fonte_resumo'], how='left')
        .assign(adotantes_mes=lambda x: np.round(x['adotantes_mes'] * x['part_fonte'], 0))
    )
    projecao = projecao.merge(historico_adot_fontes, on=['disco', 'segmento', 'ano', 'mes', 'fonte_resumo'], how='left')
    projecao = (
        projecao
        .assign(adotantes_mes=lambda x: np.where(x['date'] <= datetime(2024, 2, 1),
                                                 x['adotantes_hist'], x['adotantes_mes']))
        .groupby(['disco', 'segmento', 'fonte_resumo'])
        .apply(lambda x: x.assign(adotantes_acum=x['adotantes_mes'].cumsum()))
        .reset_index(drop=True)
        .assign(mercado_potencial=lambda x: x['mercado_potencial'] / 4)
    )

    # Cálculo percentual de adocao frente ao numero de consumidores totais
    adotantes_segmento = (
        projecao
        .assign(
            segmento=np.where(projecao['segmento'] == "residencial_remoto", "residencial", projecao['segmento']))
        .assign(
            segmento=np.where(projecao['segmento'] == "comercial_at_remoto", "comercial_bt", projecao['segmento']))
        .groupby(['disco', 'ano', 'mes', 'segmento'])
        .agg(adotantes=('adotantes_acum', 'sum'), mercado_potencial=('mercado_potencial', 'sum'))
        .reset_index()
    )

    mercado_nicho = (
        input_consumidores_nicho
        .assign(segmento=np.where(input_consumidores_nicho['segmento'] == "residencial_remoto", "residencial",
                                  input_consumidores_nicho['segmento']))
        .assign(segmento=np.where(input_consumidores_nicho['segmento'] == "comercial_at_remoto", "comercial_bt",
                                  input_consumidores_nicho['segmento']))
        .groupby(['disco', 'ano', 'mes', 'segmento'])
        .agg(mercado_nicho=('consumidores', 'sum'))
        .reset_index()
    )

    results_part_adotantes = (
        pd.merge(adotantes_segmento, input_consumidores_totais, on=['disco', 'ano', 'mes', 'segmento'])
        .assign(penetracao_total=lambda x: x['adotantes'] / x['total_ucs'])
        .merge(mercado_nicho, on=['disco', 'ano', 'mes', 'segmento'])
        .assign(penetracao_nicho=lambda x: x['adotantes'] / x['mercado_nicho'],
                penetracao_potencial=lambda x: x['adotantes'] / x['mercado_potencial'])
    )

    # Projeção
    results_proj_adotantes = projecao.copy()

    return results_proj_adotantes, results_part_adotantes


def otimiza_casos(base_otimizacao, row, p_max, q_max):
    caso = base_otimizacao[(base_otimizacao['disco'] == row['disco']) &
                           (base_otimizacao['segmento'] == row['segmento'])]

    otimizador = minimize(otimiza_curva_s, np.array([0.005, 0.3]), args=(caso,), method='L-BFGS-B',
                          bounds=[(0.0001, p_max), (0.01, q_max)], options={'maxiter': 100})

    parametros = {'disco': row['disco'], 'segmento': row['segmento'], 'p': [otimizador.x[0]], 'q': [otimizador.x[1]]}
    return parametros


def otimiza_curva_s(params, y):
    p, q = params
    spb = y['spb']
    consumidores = y['consumidores']
    a = y['ano']
    m = y['mes']
    payback = y['payback']
    historico_adotantes = y['adotantes_acum']

    # Quantidade de meses
    t = (m / 12) + ((12 * a - 12) / 12)

    # Como o payback foi calculado de forma anual, multiplicando o payback por 12 temos o retorno em meses
    taxa_difusao = (1 - np.exp(- (p + q) * t)) / (1 + (q / p) * np.exp(- (p + q) * t))
    mercado_potencial = np.exp(- spb * payback / 12) * consumidores
    projecao = taxa_difusao * mercado_potencial
    erro = np.sum((historico_adotantes - projecao) ** 2)

    return erro


def get_calibra_curva_s(results_payback, results_consumidores, p_max, q_max,
                        dir_dados_premissas):
    """Calibra o modelo de Bass com dados históricos e gera curvas S de adoção.

        Args:
            results_payback (dataframe): Resultado da função [get_payback].
            results_consumidores (dataframe): Resultado da função [get_mercado_potencial].
            p_max (float): Fator de inovação (p) máximo. Default igual a 0.01.
            q_max (float): Fator de imitação (q) máximo. DEfault igual a 1.
            dir_dados_premissas (string): Diretório onde se encontram as premissas.

        Returns:
            dataframe: curvas de difusão e mercado potencial
    """

    # Ler dados
    dados_gd = pd.read_excel(os.path.join(dir_dados_premissas, "base_mmgd.xlsx"), header=0, sheet_name='Sheet1')
    fator_sbp = pd.read_excel(os.path.join(dir_dados_premissas, "spb.xlsx"), header=0, sheet_name='fator')
    tipo_payback = pd.read_excel(os.path.join(dir_dados_premissas, "tipo_payback.xlsx"), header=0, sheet_name='tipo')

    # Agrupar casos de otimização
    casos_otimizacao = \
        results_payback.groupby(['disco', 'segmento']).size().reset_index(name='count').drop(columns=['count'])

    # Agrupar histórico
    historico = \
        dados_gd.groupby(['disco', 'segmento', 'ano', 'mes'])['num_geradores'].sum().reset_index(name='adotantes_hist')

    # Resultado_payback_historico
    resultado_payback_historico = results_payback.copy()
    resultado_payback_historico['date'] = \
        resultado_payback_historico.apply(lambda lin: datetime(lin['ano'], lin['mes'], 1), axis=1)
    resultado_payback_historico = \
        resultado_payback_historico[resultado_payback_historico['date'] <= datetime(2024, 2, 1)]
    resultado_payback_historico = resultado_payback_historico[['disco', 'segmento', 'ano',
                                                               'mes', 'payback', 'payback_desc']]

    # Colocando payback NaN para um tempo alto de retorno
    resultado_payback_historico['payback'] = resultado_payback_historico['payback'].fillna(100)
    resultado_payback_historico['payback_desc'] = resultado_payback_historico['payback_desc'].fillna(100)

    # Juntar dados
    base_otimizacao = \
        pd.merge(resultado_payback_historico, results_consumidores, on=['disco', 'ano', 'mes', 'segmento'])
    base_otimizacao = pd.merge(base_otimizacao, historico, on=['disco', 'ano', 'segmento', 'mes'])
    base_otimizacao = pd.merge(base_otimizacao, tipo_payback, on='segmento', how='left')
    base_otimizacao['adotantes_hist'] = base_otimizacao['adotantes_hist'].fillna(0)

    # Ajustar payback de acordo com o tipo_payback
    base_otimizacao['payback'] = \
        base_otimizacao.apply(lambda lin:
                              lin['payback'] if lin['tipo_payback'] == 'simples' else lin['payback_desc'], axis=1)

    # Ordenar e agrupar
    base_otimizacao['adotantes_acum'] = base_otimizacao.groupby(['disco', 'segmento'])['adotantes_hist'].cumsum()
    base_otimizacao = base_otimizacao[['disco', 'consumidores', 'ano',
                                       'mes', 'payback', 'adotantes_acum',
                                       'segmento']].assign(ano=lambda x: x['ano'] - 2012)
    base_otimizacao = pd.merge(base_otimizacao, fator_sbp, on='segmento')

    # Otimizar casos
    optimum_cases = []
    for index, row in casos_otimizacao.iterrows():
        optimum_cases.append(pd.DataFrame(otimiza_casos(base_otimizacao, row, p_max, q_max)))
    optimum_cases = pd.concat(optimum_cases, ignore_index=True)

    optimum_cases = pd.merge(optimum_cases, fator_sbp, on='segmento')

    # Anos de simulação
    anos_simulacao = pd.DataFrame({'ano': np.arange(1, 23)})
    optimum_cases = pd.merge(optimum_cases, anos_simulacao, how='cross')

    # Meses de simulação
    meses_simulacao = pd.DataFrame({'mes': np.arange(1, 13)})
    optimum_cases = pd.merge(optimum_cases, meses_simulacao, how='cross')

    # Calcular Ft
    optimum_cases['Ft'] = \
        (1 - np.exp(-(optimum_cases['p'] + optimum_cases['q']) *
                    optimum_cases['ano'])) / (1 + (optimum_cases['q'] / optimum_cases['p']) *
                                              np.exp(-(optimum_cases['p'] + optimum_cases['q']) * optimum_cases['ano']))

    # Calcular ano
    optimum_cases['ano'] = optimum_cases['ano'] + 2012

    # Juntar com consumidores
    optimum_cases = pd.merge(optimum_cases, results_consumidores, on=['disco', 'segmento', 'ano', 'mes'])

    # Juntar com casos_otimizados
    optimum_cases = pd.merge(optimum_cases, results_payback, on=['disco', 'segmento', 'ano', 'mes'])

    # Calcular mercado_potencial
    optimum_cases['payback'] = optimum_cases['payback'].fillna(100000)
    optimum_cases['mercado_potencial'] = \
        np.round(np.exp(-optimum_cases['spb'] * optimum_cases['payback']) * optimum_cases['consumidores'], 0)
    optimum_cases['mercado_potencial'] = np.where(optimum_cases['mercado_potencial'] == 0, 1,
                                                  optimum_cases['mercado_potencial'])

    return optimum_cases


def get_mercado_potencial(ano_base, tx_cresc_grupo_a, filtro_renda_domicilio, fator_local_comercial,
                          dir_dados_premissas, ano_max_resultado):
    """Cria a base do mercado potencial inicial para a adoção.

        Args:
            ano_base (int): Ano base da projeção. Define o ano em que a função irá buscar a base de dados. Último ano
                                completo realizado.
            tx_cresc_grupo_a (float): Taxa de crescimento anual dos consumuidores cativos do Grupo A.
            filtro_renda_domicilio (string): Define o filtro aplicado a consumidores residenciais, de acordo com a
                                                 renda mensal do responsável, em salários mínimos. Permite: "total",
                                                 "maior_1sm, maior_2sm", "maior_3sm" ou "maior_5sm". Default igual a
                                                 "maior_3sm".
            fator_local_comercial (string): Define a origem dos dados do Fator de Aptidão Local "FAL" para os
                                                consumidores não residenciais atendidos em baixa tensão. Como default,
                                                são utilizados os mesmos valores dos consumidores residenciais. Caso
                                                selecionado "historico", utiliza o histórico do percentual de adotantes
                                                locais por distribuidora até o ano base.
            dir_dados_premissas (string): Diretório onde se encontram as premissas.
            ano_max_resultado (int): Ano final para apresentação dos resultados. Máximo igual a 2060. Default igual
                                        a 2060.

        Returns:
            list: lista com dois dataframes. "consumidores" possui o mercado potencial incial.
                      "consumidores_totais" possui dados de mercado total.
        """

    # Total de domicílios
    total_domicilios = pd.read_excel(os.path.join(dir_dados_premissas, "total_domicilios.xlsx"),
                                     header=0,
                                     sheet_name='Planilha1')

    # Residencial
    crescimento_mercado = pd.read_excel(os.path.join(dir_dados_premissas, "crescimento_mercado.xlsx"), header=0,
                                        sheet_name='Planilha1')
    crescimento_mercado['crescimento_acumulado'] = \
        crescimento_mercado.groupby('disco')['taxa_crescimento_mercado'].apply(lambda x: (1 + x).cumprod())
    crescimento_mercado.drop('taxa_crescimento_mercado', axis=1, inplace=True)

    # Consumidores residenciais
    consumidores_residenciais = pd.read_excel(os.path.join(dir_dados_premissas, "consumidores_residenciais_renda.xlsx"),
                                              header=0,
                                              sheet_name='Sheet1')
    consumidores_residenciais['maior_5sm'] = \
        consumidores_residenciais['domicilios_5a10sm'] + consumidores_residenciais['domicilios_10a15sm'] + \
        consumidores_residenciais['domicilios_15a20sm'] + consumidores_residenciais['domicilios_maior20sm']
    consumidores_residenciais['maior_3sm'] = \
        consumidores_residenciais['maior_5sm'] + consumidores_residenciais['domicilios_3a5sm']
    consumidores_residenciais['maior_2sm'] = \
        consumidores_residenciais['maior_3sm'] + consumidores_residenciais['domicilios_2a3sm']
    consumidores_residenciais['maior_1sm'] = \
        consumidores_residenciais['maior_2sm'] + consumidores_residenciais['domicilios_1a2sm']
    consumidores_residenciais = consumidores_residenciais[['disco', 'maior_5sm', 'maior_3sm', 'maior_2sm', 'maior_1sm']]
    consumidores_residenciais = \
        pd.melt(
            consumidores_residenciais,
            id_vars=['disco'],
            value_vars=['maior_5sm', 'maior_3sm', 'maior_2sm', 'maior_1sm'],
            var_name='renda',
            value_name='domicilios'
        )

    # Lista os consumidores residenciais
    lista_consumidores_residenciais = \
        consumidores_residenciais.groupby(['disco', 'renda']).size().reset_index(name='count')
    lista_consumidores_residenciais = lista_consumidores_residenciais.drop(columns=['count'])
    lista_consumidores_residenciais = lista_consumidores_residenciais.reset_index(drop=True)
    anos_faltantes_res = pd.DataFrame({'ano': range(2013, ano_max_resultado + 1)})
    lista_consumidores_residenciais = \
        pd.merge(
            lista_consumidores_residenciais.assign(key=1),
            anos_faltantes_res.assign(key=1),
            on='key').drop(columns='key')
    consumidores_residenciais = \
        pd.merge(
            lista_consumidores_residenciais,
            consumidores_residenciais,
            on=['disco', 'renda']
        )
    consumidores_residenciais = pd.merge(consumidores_residenciais, crescimento_mercado, on=['ano', 'disco'])

    # Acrescentando projeção de consumidores residenciais
    consumidores_residenciais['consumidores_proj'] = \
        consumidores_residenciais['domicilios'] * consumidores_residenciais['crescimento_acumulado']
    consumidores_residenciais['consumidores_proj'] = consumidores_residenciais['consumidores_proj'].astype(int)
    consumidores_residenciais = consumidores_residenciais[['disco', 'ano', 'mes', 'renda', 'consumidores_proj']]
    consumidores_residenciais = consumidores_residenciais[consumidores_residenciais['ano'] > 2012]

    # Consumidores B2 e B3
    consumidores_b2b3 = pd.read_excel(os.path.join(dir_dados_premissas, "consumidores_b2b3.xlsx"), header=0,
                                      sheet_name='Sheet1')
    consumidores_b2b3 = pd.melt(consumidores_b2b3,
                                id_vars=['disco'],
                                value_vars=['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
                                            '2022', '2023'],
                                var_name='ano',
                                value_name='consumidores')
    consumidores_b2b3['ano'] = consumidores_b2b3['ano'].astype(int)
    consumidores_b2b3 = consumidores_b2b3.groupby(['disco', 'ano'])['consumidores'].sum().reset_index()
    consumidores_b2b3 = consumidores_b2b3[(consumidores_b2b3['ano'] == ano_base)]

    # Lista de consumidores
    lista_consumidores_b2b3 = consumidores_b2b3[['disco']].drop_duplicates()
    lista_consumidores_b2b3 = \
        pd.merge(
            lista_consumidores_b2b3.assign(key=1),
            anos_faltantes_res.assign(key=1),
            on='key').drop(columns='key')
    consumidores_b2b3.drop(['ano'], axis=1, inplace=True)
    consumidores_b2b3 = \
        pd.merge(
            lista_consumidores_b2b3,
            consumidores_b2b3,
            on=['disco']
        )
    consumidores_b2b3 = consumidores_b2b3[consumidores_b2b3['ano'] > 2012]
    consumidores_b2b3 = pd.merge(consumidores_b2b3, crescimento_mercado, on=['disco', 'ano'])

    # Acrescentando projeção de consumidores B2 e B3
    consumidores_b2b3['consumidores_proj'] = \
        consumidores_b2b3['consumidores'] * consumidores_b2b3['crescimento_acumulado']
    consumidores_b2b3['consumidores_proj'] = consumidores_b2b3['consumidores_proj'].astype(int)
    consumidores_b2b3 = consumidores_b2b3[['disco', 'ano', 'mes', 'consumidores_proj']]

    # Consumidores grupo A
    consumidores_a = pd.read_excel(os.path.join(dir_dados_premissas, "consumidores_a.xlsx"),
                                   header=0,
                                   sheet_name='Sheet1')
    consumidores_a = pd.melt(consumidores_a,
                             id_vars=['disco'],
                             value_vars=['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
                                         '2022', '2023'],
                             var_name='ano',
                             value_name='consumidores')
    consumidores_a['ano'] = consumidores_a['ano'].astype(int)
    consumidores_a = consumidores_a.groupby(['disco', 'ano'])['consumidores'].sum().reset_index()
    consumidores_a = consumidores_a[consumidores_a['ano'] == ano_base]

    # Lista de consumidores
    consumidores_a.drop(['ano'], axis=1, inplace=True)
    consumidores_a = pd.merge(lista_consumidores_b2b3, consumidores_a, on=['disco'])
    consumidores_a = consumidores_a[consumidores_a['ano'] > 2012]
    consumidores_a['crescimento'] = [1 + tx_cresc_grupo_a] * len(consumidores_a)
    consumidores_a['taxa_acumulada'] = np.cumprod(consumidores_a['crescimento'].tolist()).tolist()
    consumidores_a['consumidores_proj'] = consumidores_a['consumidores'] * consumidores_a['taxa_acumulada']
    consumidores_a['consumidores_proj'] = consumidores_a['consumidores_proj'].astype(int)
    consumidores_a = consumidores_a[['disco', 'ano', 'consumidores_proj']]

    # Acrescentando mês
    meses = range(1, 13)
    combinacoes = [(ano, mes) for ano in consumidores_a['ano'].unique() for mes in meses]
    df_combinacoes = pd.DataFrame(combinacoes, columns=['ano', 'mes'])
    consumidores_a = df_combinacoes.merge(consumidores_a, on='ano', how='left')

    # Consumidores totais para avaliação de share posterior
    consumidores_totais_domicilios = total_domicilios.copy()
    consumidores_totais_domicilios['segmento'] = ["residencial"] * len(consumidores_totais_domicilios)
    consumidores_totais_domicilios.rename(columns={'domicilios': 'total_ucs'}, inplace=True)

    consumidores_totais_b2b3 = \
        consumidores_b2b3.groupby(['ano', 'mes', 'disco']).agg(total_ucs=('consumidores_proj', 'sum')).reset_index()
    consumidores_totais_b2b3['segmento'] = ["comercial_bt"] * len(consumidores_totais_b2b3)

    consumidores_totais_a = \
        consumidores_a.groupby(['ano', 'mes', 'disco']).agg(total_ucs=('consumidores_proj', 'sum')).reset_index()
    consumidores_totais_a['segmento'] = ["comercial_at"] * len(consumidores_totais_a)

    consumidores_totais_todos = pd.concat([consumidores_totais_domicilios, consumidores_totais_b2b3,
                                           consumidores_totais_a],
                                          ignore_index=True)

    # Calculo mercado nicho
    fator_tecnico = pd.read_excel(os.path.join(dir_dados_premissas, "fator_tecnico.xlsx"),
                                  header=0,
                                  sheet_name='Sheet1')
    fator_tecnico_comercial = fator_tecnico.copy()

    if fator_local_comercial == "historico":
        dados_gd = pd.read_excel(os.path.join(dir_dados_premissas, "base_mmgd.xlsx"), header=0, sheet_name='Sheet1')
        fator_tecnico_comercial = (
            dados_gd[(dados_gd['segmento'] == "comercial_bt")]
            .groupby(['disco', 'local_remoto'])
            .agg(qtde_clientes=('qtde_u_csrecebem_os_creditos', 'sum'))
            .reset_index()
            .groupby('disco')
            .apply(lambda x: x.assign(total_clientes=x['qtde_clientes'].sum()))
            .reset_index(drop=True)
            .pipe(lambda x: x[x['local_remoto'] == "local"])
            .assign(fator_tecnico=lambda x: x['qtde_clientes'] / x['total_clientes'])
        )

    consumidores_residenciais = consumidores_residenciais[consumidores_residenciais['renda'] == filtro_renda_domicilio]

    fator_comercial = \
        consumidores_residenciais.groupby(['disco', 'ano', 'mes']).agg(consumidores_nicho=('consumidores_proj',
                                                                                           'sum')).reset_index()
    fator_comercial = pd.merge(fator_comercial, total_domicilios, on=['ano', 'mes', 'disco'], how='left')
    fator_comercial['fator_nicho_comercial'] = fator_comercial['consumidores_nicho'] / fator_comercial['domicilios']
    fator_comercial = fator_comercial[['ano', 'mes', 'fator_nicho_comercial']]

    consumidores_residenciais = pd.merge(consumidores_residenciais, fator_tecnico, on='disco')
    consumidores_residenciais['residencial'] = \
        consumidores_residenciais['consumidores_proj'] * consumidores_residenciais['fator_tecnico']
    consumidores_residenciais['residencial'] = consumidores_residenciais['residencial'].astype(int)
    consumidores_residenciais['residencial_remoto'] = \
        consumidores_residenciais['consumidores_proj'] * (1 - consumidores_residenciais['fator_tecnico'])
    consumidores_residenciais['residencial_remoto'] = consumidores_residenciais['residencial_remoto'].astype(int)
    consumidores_residenciais = consumidores_residenciais[['disco', 'ano', 'mes', 'residencial', 'residencial_remoto']]
    consumidores_residenciais = pd.melt(consumidores_residenciais, id_vars=['disco', 'ano', 'mes'],
                                        value_vars=['residencial', 'residencial_remoto'], var_name='segmento',
                                        value_name='consumidores')

    consumidores_b2b3 = pd.merge(consumidores_b2b3, fator_comercial, on=['ano', 'mes'])
    consumidores_b2b3 = pd.merge(consumidores_b2b3, fator_tecnico_comercial, on='disco')
    consumidores_b2b3['comercial_bt'] = \
        consumidores_b2b3['consumidores_proj'] * consumidores_b2b3['fator_tecnico'] * \
        consumidores_b2b3['fator_nicho_comercial']
    consumidores_b2b3['comercial_bt'] = consumidores_b2b3['comercial_bt'].astype(int)
    consumidores_b2b3['comercial_at_remoto'] = \
        consumidores_b2b3['consumidores_proj'] * consumidores_b2b3['fator_nicho_comercial'] * \
        (1 - consumidores_b2b3['fator_tecnico'])
    consumidores_b2b3['comercial_at_remoto'] = consumidores_b2b3['comercial_at_remoto'].astype(int)
    consumidores_b2b3 = consumidores_b2b3[['disco', 'ano', 'mes', 'comercial_bt', 'comercial_at_remoto']]
    consumidores_b2b3 = pd.melt(consumidores_b2b3, id_vars=['disco', 'ano', 'mes'],
                                value_vars=['comercial_bt', 'comercial_at_remoto'], var_name='segmento',
                                value_name='consumidores')

    consumidores_a['segmento'] = ["comercial_at"] * len(consumidores_a)
    consumidores_a.rename(columns={'consumidores_proj': 'consumidores'}, inplace=True)
    consumidores_a = consumidores_a[['disco', 'ano', 'mes', 'segmento', 'consumidores']]

    # Lista com os dataframes de resultados
    consumidores_nicho_todos = pd.concat([consumidores_residenciais, consumidores_b2b3, consumidores_a],
                                         ignore_index=True)

    return consumidores_nicho_todos, consumidores_totais_todos


def get_payback(casos, premissas_reg, inflacao, taxa_desconto_nominal, ano_troca_inversor,
                pagamento_disponibilidade, disponibilidade_kwh_mes, desconto_capex_local, anos_desconto,
                dir_dados_premissas, vida_util, ano_max_resultado):
    """Roda um fluxo de caixa para cada caso e retorna métricas financeiras.

        Args:
            casos (dataframe): Base gerada pela função [get_casos_payback]
            premissas_reg (dataframe): Input de premissas regulatórias para serem consideradas nos cálculos.
            inflacao (float): Taxa anual de inflacao considerada no reajuste das tarifas e para calcular o retorno real
                              de projetos.
            taxa_desconto_nominal (float): Taxa de desconto nominal considerada nos cálculos de payback descontado.
                                           Default igual a 0.13.
            ano_troca_inversor (int): Ano, a partir do ano de instalação, em que é realizada a troca do inversor
                                      fotovoltaico.
            pagamento_disponibilidade (float): Percentual de meses em que o consumidor residencial paga custo de
                                               disponbilidade em função da variabilidade da geração fotovoltaica.
                                               Tem efeito somente até o ano de 2022.
            disponibilidade_kwh_mes (float): Consumo de disponbilidade do consumidor em kWh/mês. Default igual a 100,
                                            equivalente a um consumidor trifásico. Tem efeito somente até o ano de 2022.
            desconto_capex_local (float): Percentual de desconto a ser aplicado no CAPEX de sistemas de geração
                                          local (ex: 0.1) para simulação de incentivos.
            anos_desconto (list): Anos em que há a incidência do desconto no CAPEX. Ex: c(2024, 2025). Caso não se
                                  aplique, informar 0.
            dir_dados_premissas (string): Diretório onde se encontram as premissas.
            vida_util (int): Vida útil do sistema fotovoltaico em anos.
            ano_max_resultado (int): Ano final para apresentação dos resultados. Máximo igual a 2060. Default igual a
                                     2060.

        Returns:
            pd.DataFrame: Métricas financeiras para cada caso.
    """

    # Fator de Construção do investimento
    fator_construcao = pd.read_excel(os.path.join(dir_dados_premissas, "tempo_construcao.xlsx"),
                                     header=0, sheet_name='fator')

    # Tarifas
    tarifas = get_tarifas(pd.read_excel(os.path.join(dir_dados_premissas, "tarifas.xlsx"),
                                        header=0, sheet_name='Sheet1'), ano_max_resultado)

    # Premissas Regulatorias
    premissas_reg['alternativa'] = \
        premissas_reg.apply(
            lambda x: 2 if x['binomia'] and x['alternativa'] in [0, 1] else x['alternativa'], axis=1
        )

    # Lista para armazenar os DataFrames resultantes de cada iteração do fluxo de caixa
    total_results = []

    # Calculando o fluxo de caixa para cada caso
    for index, row in casos.iterrows():
        if row['mes'] == 1:
            results = get_fluxo_de_caixa(row, vida_util, disponibilidade_kwh_mes, inflacao, ano_troca_inversor,
                                         desconto_capex_local, anos_desconto, pagamento_disponibilidade,
                                         taxa_desconto_nominal, fator_construcao, tarifas, premissas_reg)
            total_results.append(results)

    # Empilhar todos os DataFrames em um único DataFrame final
    results_payback = pd.concat(total_results, ignore_index=True)

    # Concatenando o DataFrame de paybacks ao DataFrame existente de casos
    results_payback = pd.merge(casos, results_payback, on=['disco', 'segmento', 'ano'])

    return results_payback


def get_casos_payback(ano_max_resultado, inflacao, ano_troca_inversor, fator_custo_inversor, dir_dados_premissas):
    """Cria a base de casos para serem simulados posteriormente no cálculo do payback.

        Args:
            ano_max_resultado (int): Ano final para apresentação dos resultados.
            inflacao (float): Taxa anual de inflacao considerada no reajuste das tarifas e para calcular o retorno real
                              de projetos.
            ano_troca_inversor (int): Ano, a partir do ano de instalação, em que é realizada a troca do inversor
                                      fotovoltaico.
            fator_custo_inversor (float): Custo do inversor para a substituição em percentual do CAPEX total do sistema
                                          de geração.
            dir_dados_premissas (string): Diretório onde se encontram as premissas.

        Returns:
            pd.DataFrame: Casos para serem simulados posteriormente no cálculo do payback.
    """

    # Potência típica dos sistemas
    potencia_tipica = pd.read_excel(os.path.join(dir_dados_premissas, "potencia_tipica.xlsx"),
                                    header=0, sheet_name='Sheet1')

    # Fator de capacidade
    fc_fontes = pd.read_excel(os.path.join(dir_dados_premissas, "fc_distribuidoras.xlsx"),
                              header=0, sheet_name='Sheet1')

    # Fator de autoconsumo
    injecao = pd.read_excel(os.path.join(dir_dados_premissas, "injecao.xlsx"), header=0, sheet_name='Sheet1')
    injecao['oem_anual'] = injecao['segmento'].map(lambda seg: 0.01 if seg != "comercial_at_remoto" else 0.02)

    # Dados de UFV
    ufv_fonte = {'fonte_resumo': ["Fotovoltaica"],
                 'vida_util': [20],
                 'degradacao': [0.005]}
    ufv_fonte = pd.DataFrame(ufv_fonte)

    # Custos
    custos = pd.read_excel(os.path.join(dir_dados_premissas, "custos.xlsx"), header=0, sheet_name='custos')
    custos['custo_inversor'] = custos['custo_unitario'].map(lambda cus: cus * fator_custo_inversor)

    # Casos
    casos = pd.merge(injecao, fc_fontes, on='disco')
    casos['fv'] = [casos['fc_mini'].iloc[row]
                   if casos['segmento'].iloc[row] in ['comercial_at', 'comercial_at_remoto']
                   else casos['fc_micro'].iloc[row]
                   for row in range(len(casos))]
    casos.drop(['fc_mini', 'fc_micro'], axis=1, inplace=True)
    casos.rename(columns={'fv': 'fc'}, inplace=True)
    casos = pd.merge(casos, ufv_fonte, on='fonte_resumo')
    casos = pd.merge(casos, potencia_tipica, on=['disco', 'segmento'])

    # Unindo casos com o custo
    casos = pd.merge(casos, custos, on='segmento')

    # Casos para o payback
    casos['capex_inicial'] = [casos['custo_unitario'].iloc[row] * casos['pot_sistemas_kw'].iloc[row] * 1000
                              for row in range(len(casos))]
    casos['capex_inversor'] = [casos['custo_inversor'].iloc[row] * casos['pot_sistemas_kw'].iloc[row] * 1000 *
                               ((1 + inflacao) ** (ano_troca_inversor - 1)) for row in range(len(casos))]
    casos['geracao_kwh_ano'] = \
        [casos['pot_sistemas_kw'].iloc[row] * casos['fc'].iloc[row] * 360 * 24 for row in
         range(len(casos))
         ]

    # Filtrando período de interesse
    casos = casos[casos['ano'] <= ano_max_resultado]

    return casos


def get_parametros():
    """Define os parãmetros utilizados para a projeção

        Input:
            ano_base (int): Ano base da projeção. Define o ano em que a função irá buscar a base de dados. Último ano
                            completo realizado.
            ano_max_resultado (int): Ano final para apresentação dos resultados.
            altera_sistemas_existentes (boolean): True se alterações regulatórias afetam investimentos realizados em
                                                  anos anteriores à revisão da regulação
            ano_decisao_alteracao (int): Ano em que são definidas novas regras e se tornam de conhecimento público.
                                             Esse parâmetro só tem efeito caso altera_sistemas_existentes  seja igual a
                                             True. Default igual a 2023.
            inflacao (float): Taxa anual de inflacao considerada no reajuste das tarifas e para calcular o retorno real
                              de projetos.
            fator_custo_inversor (float): Custo do inversor para a substituição em percentual do CAPEX total do sistema
                                          de geração.
            taxa_desconto_nominal (float): Taxa de desconto nominal considerada nos cálculos de payback descontado.
                                           Default igual a 0.13.
            custo_reforco_rede (float): Custo em R$/kW aplicado a projetos de geracao remota em Alta Tensão. Representa
                                        um custo pago pelo empreendedor para reforços na rede. Default igual a 200.
            ano_troca_inversor (int): Ano, a partir do ano de instalação, em que é realizada a troca do inversor
                                      fotovoltaico.
            pagamento_disponibilidade (float): Percentual de meses em que o consumidor residencial paga custo de
                                               disponbilidade em função da variabilidade da geração fotovoltaica.
                                               Tem efeito somente até o ano de 2022.
            disponibilidade_kwh_mes (float): Consumo de disponbilidade do consumidor em kWh/mês. Default igual a 100,
                                            equivalente a um consumidor trifásico. Tem efeito somente até o ano de 2022.
            filtro_renda_domicilio (string): Define o filtro aplicado a consumidores residenciais, de acordo com a renda
                                             mensal do responsável, em salários mínimos. Permite: "total", "maior_1sm",
                                             maior_2sm", "maior_3sm" ou "maior_5sm".
            fator_local_comercial (string): Define a origem dos dados do Fator de Aptidão Local "FAL" para os
                                            consumidores não residenciais atendidos em baixa tensão. Como default, são
                                            utilizados os mesmos valores dos consumidores residenciais. Caso selecionado
                                            "historico", utiliza o histórico do percentual de adotantes locais por
                                            distribuidora até o ano base.
            desconto_capex_local (float): Percentual de desconto a ser aplicado no CAPEX de sistemas de geração
                                          local (ex: 0.1) para simulação de incentivos.
            anos_desconto (list): Anos em que há a incidência do desconto no CAPEX. Ex: c(2024, 2025). Caso não se
                                  aplique, informar 0.
            filtro_comercial (float): Fator percentual para definir o nicho do segmento comercial. Default é calculado
                                      pelo modelo com base no nicho residencial.
            tx_cresc_grupo_a (float): Taxa de crescimento anual dos consumuidores cativos do Grupo A.
            p_max (float): Fator de inovação (p) máximo.
            q_max (float): Fator de imitação (q) máximo.
            ajuste_ano_corrente (boolean): Se True indica que a projeção deverá incorporar o histórico mensal recente,
                                           verificado em parte do primeiro ano após o ano base. Default igual a False.
                                           O arquivo base_mmgd.xlsx deve incorporar esse histórico.
            ultimo_mes_ajuste (int): Último mês com dados completos na base_ano_corrente. Só tem efeito caso
                                     ajuste_ano_corrente seja igual a True.
            metodo_ajuste (string): Se igual a "extrapola" o modelo irá extrapolar a potência e o número de adotantes
                                    até o final do ano base + 1 com base no verificado até o ultimo_mes_ajuste. Só tem
                                    efeito caso ajuste_ano_corrente seja igual a True.
            dir_dados_premissas (string): Diretório onde se encontram as premissas.

        Returns:
            dict: dicionário com parâmetros de entrada
    """

    dict_args = {
        'ano_base': 2023,
        'ano_max_resultado': 2034,
        'altera_sistemas_existentes': False,
        'ano_decisao_alteracao': 2023,
        'inflacao': 0.0375,
        'fator_custo_inversor': 0.15,
        'taxa_desconto_nominal': 0.13,
        'custo_reforco_rede': 200,
        'ano_troca_inversor': 11,
        'pagamento_disponibilidade': 0.4,
        'disponibilidade_kwh_mes': 100,
        'filtro_renda_domicilio': "maior_3sm",
        'fator_local_comercial': "historico",
        'desconto_capex_local': 0,
        'anos_desconto': [0],
        'filtro_comercial': 0.07,
        'tx_cresc_grupo_a': 0,
        'p_max': 0.01,
        'q_max': 1,
        'ajuste_ano_corrente': True,
        'ultimo_mes_ajuste': 2,
        'metodo_ajuste': "extrapola",
        'dir_dados_premissas': "./input/"
    }

    return dict_args


def get_premissas():
    """Premissas regulatórias para serem consideradas nos cálculos

        Input:
            ano (int): diretório onde os dados observados de temperatura do SaMET estão armazenados
            alternativa (int):
                              + 0: Consumidor compensa todas as componentes tarifárias;
                              + 1: Paga TUSD Distribuição;
                              + 2: Anterior + TUSD Transmissão.
                              + 3: Anterior + TUSD Encargos.
                              + 4: Anterior + TUSD Perdas.
                              + 5: Anterior + TE Encargos. Ou seja, compensa somente a TE Energia.
            p_transicao (float): Parcela do custo da alternativa escolhida no parâmetro alternativa a ser pago pelo
                                 consumidor
            binomia (boolean): Define se há cobrança de uma tarifa binômia na baixa tensão, em que as componentes TUSD
                               Distribuição e TUSD Transmissão passariam a ser cobradas de forma fixa, não sendo
                               passíveis de compensação
            demanda_g (boolean): Define se há cobrança de TUSDg para a demanda de consumidores do grupo A. Caso seja
                                 `FALSE`, é considerada a cobrança da TUSD consumo.

        Returns:
            pd.DataFrame: DataFrame de premissas reguratórias
        """

    return pd.read_excel("./input/premissas_reg.xlsx", header=0, sheet_name='Sheet1')


if __name__ == '__main__':
    # Carrega parâmetros para o modelo
    dict_parametros = get_parametros()

    # Carrega as premissas regutarórias ao longo dos próximos anos
    df_premissas_reg = get_premissas()

    # Construindo casos para o payback
    casos_payback = get_casos_payback(ano_max_resultado=dict_parametros['ano_max_resultado'],
                                      inflacao=dict_parametros['inflacao'],
                                      ano_troca_inversor=dict_parametros['ano_troca_inversor'],
                                      fator_custo_inversor=dict_parametros['fator_custo_inversor'],
                                      dir_dados_premissas=dict_parametros['dir_dados_premissas']
                                      )

    # Mercado potencial
    consumidores_nicho, consumidores_totais = get_mercado_potencial(ano_base=dict_parametros['ano_base'],
                                                                    tx_cresc_grupo_a=dict_parametros[
                                                                        'tx_cresc_grupo_a'],
                                                                    filtro_renda_domicilio=dict_parametros[
                                                                        'filtro_renda_domicilio'],
                                                                    fator_local_comercial=dict_parametros[
                                                                        'fator_local_comercial'],
                                                                    dir_dados_premissas=dict_parametros[
                                                                        'dir_dados_premissas'],
                                                                    ano_max_resultado=dict_parametros[
                                                                        'ano_max_resultado']
                                                                    )

    # Cálculo do Payback
    resultado_payback = get_payback(casos=casos_payback,
                                    premissas_reg=df_premissas_reg,
                                    inflacao=dict_parametros['inflacao'],
                                    taxa_desconto_nominal=dict_parametros['taxa_desconto_nominal'],
                                    ano_troca_inversor=dict_parametros['ano_troca_inversor'],
                                    pagamento_disponibilidade=dict_parametros['pagamento_disponibilidade'],
                                    disponibilidade_kwh_mes=dict_parametros['disponibilidade_kwh_mes'],
                                    desconto_capex_local=dict_parametros['desconto_capex_local'],
                                    anos_desconto=dict_parametros['anos_desconto'],
                                    dir_dados_premissas=dict_parametros['dir_dados_premissas'],
                                    vida_util=20,
                                    ano_max_resultado=dict_parametros['ano_max_resultado']
                                    )

    # Calibração da curva S
    casos_otimizados = get_calibra_curva_s(results_payback=resultado_payback,
                                           results_consumidores=consumidores_nicho,
                                           p_max=dict_parametros['p_max'],
                                           q_max=dict_parametros['q_max'],
                                           dir_dados_premissas=dict_parametros['dir_dados_premissas']
                                           )

    # Projeção de adotantes
    proj_adotantes, part_adotantes = get_proj_adotantes(results_casos_otm=casos_otimizados,
                                                        input_consumidores_totais=consumidores_totais,
                                                        input_consumidores_nicho=consumidores_nicho,
                                                        dir_dados_premissas=dict_parametros['dir_dados_premissas']
                                                        )
    proj_adotantes.to_excel("proj_adotantes.xlsx", index=0)
    part_adotantes.to_excel("part_adotantes.xlsx", index=0)

    # Projeção de potência
    df_potencia = get_proj_potencia(lista_adotantes=[proj_adotantes, part_adotantes],
                                    dir_dados_premissas=dict_parametros['dir_dados_premissas']
                                    )
    df_potencia.to_excel("proj_potencia.xlsx", index=0)
