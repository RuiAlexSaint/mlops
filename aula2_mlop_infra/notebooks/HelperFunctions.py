"""
HelperFunctions
==========

A collection of reusable utility functions for data processing, analysis,
and database operations.

Author: Rui Santos
Date: [June 2025]
Version: 1.0

Dependencies:
- pandas
- numpy
- re
- datetime
- matplotlib/seaborn (optional, for plotting functions)
- mysql.connector
- sklearn
"""

import pandas as pd
import numpy as np
import re
import logging
import os
from typing import Union, List, Dict, Any, Optional, Tuple
from datetime import time as dt_time, datetime, timedelta
import mysql.connector
from mysql.connector import Error
from mysql.connector.cursor import MySQLCursorDict
import openmeteo_requests
import requests_cache
from retry_requests import retry
from math import radians, sin, cos, sqrt, atan2
import time
import unicodedata
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import warnings
from typing import Literal


class DataCleaning:
    """
    Collection of functions for data cleaning and preprocessing.
    """

    @staticmethod
    def convert_to_float(valor: Any) -> Optional[float]:
        """
        Converte valores de string para float, tratando:
        - ',' como separador de milhares
        - '.' como separador decimal
        Formato esperado: 1,234,567.89

        Retorna:
            float ou None
        """
        # Trata casos triviais e valores não numéricos
        if pd.isna(valor):
            return None

        if not isinstance(valor, str):
            try:
                return float(valor)
            except (ValueError, TypeError):
                return None

        # Limpa a string
        valor_limpo = re.sub(r'[^\d,.-]', '', str(valor).strip().replace(' ', ''))
        if not valor_limpo:
            return None

        # Identifica formato
        if ',' in valor_limpo and '.' in valor_limpo:
            # Formato típico americano: 1,234,567.89
            if valor_limpo.rfind('.') > valor_limpo.rfind(','):
                valor_limpo = valor_limpo.replace(',', '')
            else:
                # vírgula após ponto → formato ambíguo
                return None

        elif ',' in valor_limpo and '.' not in valor_limpo:
            # Pode ser separador de milhares
            partes = valor_limpo.split(',')
            if len(partes) > 2:
                # Múltiplas vírgulas → validar padrão 3 dígitos
                if not (all(len(p) == 3 for p in partes[1:]) and len(partes[0]) <= 3):
                    return None
            # Remove todas as vírgulas (tratando como separadores de milhares)
            valor_limpo = valor_limpo.replace(',', '')

        # Se tiver apenas ponto decimal, mantém como está

        # Tenta converter
        try:
            return float(valor_limpo)
        except ValueError:
            return None

    @staticmethod
    def convert_to_int(valor: Any, arredondar: bool = True) -> Optional[int]:
        """
        Converte valores para inteiro, usando convert_to_float internamente.

        Args:
            valor: Valor de entrada (str, float, int, etc.).
            arredondar: Se True, arredonda com np.round(); se False, trunca (int()).

        Retorna:
            int ou None
        """
        val_float = DataCleaning.convert_to_float(valor)
        if val_float is None:
            return None

        try:
            if arredondar:
                return int(np.round(val_float))
            else:
                return int(val_float)  # truncar
        except (ValueError, TypeError):
            return None
        
    @staticmethod
    def clean_text_series(series: pd.Series,
                          remove_special_chars: bool = True,
                          to_lowercase: bool = False,
                          strip_whitespace: bool = True,
                          remove_non_ascii: bool = True,
                          normalize_spaces: bool = False, # Nota: Deixei False por padrão.
                          remove_diacritics: bool = False) -> pd.Series:
        """
        Clean a pandas Series (column) by applying text transformations.

        Args:
            series: pandas Series to clean.
            remove_special_chars: Whether to remove special characters.
            to_lowercase: Whether to convert to lowercase.
            strip_whitespace: Whether to strip leading/trailing whitespace.
            remove_non_ascii: Whether to remove non-ASCII characters.
            normalize_spaces: Whether to normalize spaces by replacing all internal 
                              spaces with '_' and removing leading/trailing spaces.
            remove_diacritics: Whether to remove diacritic marks (accents) by replacing them 
                                with their base letter (e.g., 'ã' -> 'a', 'ç' -> 'c'). 
                                This is automatically set to True if remove_non_ascii is True.

        Returns:
            pd.Series: Cleaned pandas Series.
        """
        cleaned_series = series.astype(str)

        # ------------------------------------------------------------------------
        # 0. Lógica de Ativação Forçada para PREVENIR a perda de acentos
        should_remove_diacritics = remove_diacritics or remove_non_ascii or remove_special_chars
        # ------------------------------------------------------------------------

        # 1. Remover espaços em branco iniciais/finais (Executado se 'strip_whitespace' for True)
        if strip_whitespace or normalize_spaces: 
            # O 'strip' é necessário para 'normalize_spaces' funcionar corretamente, 
            # garantindo que os espaços a substituir são apenas os internos.
            cleaned_series = cleaned_series.str.strip()

        # 2. Transliterar Diacríticos (Acentos)
        if should_remove_diacritics:
            def transliterate_accents(text):
                if pd.isna(text):
                    return text
                try:
                    normalized = unicodedata.normalize('NFD', str(text))
                    return "".join([c for c in normalized if not unicodedata.combining(c)])
                except TypeError:
                    return text 

            cleaned_series = cleaned_series.map(transliterate_accents)

        # 3. Remover Caracteres Não-ASCII
        if remove_non_ascii:
            cleaned_series = cleaned_series.str.replace(r'[^\x00-\x7F]+', '', regex=True)

        # 4. Normalizar Espaços
        if normalize_spaces:
            # Primeiro, consolida múltiplos espaços (incluindo tab/newline) em um único espaço.
            cleaned_series = cleaned_series.str.replace(r'\s+', ' ', regex=True)
            # Segundo, substitui todos os espaços restantes por um underscore '_'.
            cleaned_series = cleaned_series.str.replace(' ', '_', regex=False)
            
        # 5. Remover Caracteres Especiais (Pontuação)
        if remove_special_chars:
            # Se 'normalize_spaces' foi ativado, o regex deve incluir o '_' como caractere válido.
            # Caso contrário, o '_' seria removido juntamente com a pontuação.
            if normalize_spaces:
                 # ^[\w\s] permite letras, números e espaços (e '_' por causa do \w)
                 # Mas como os espaços já são '_', vamos permitir \w e '_'
                cleaned_series = cleaned_series.str.replace(r'[^\w_]', '', regex=True)
            else:
                # Comportamento padrão: apenas remove pontuação.
                cleaned_series = cleaned_series.str.replace(r'[^\w\s]', '', regex=True)

        # 6. Converter para Minúsculas
        if to_lowercase:
            cleaned_series = cleaned_series.str.lower()

        return cleaned_series

    @staticmethod
    def clean_text_value(valor: str,
                         remove_non_ascii: bool = True,
                         normalize_spaces: bool = True) -> str:
        """
        Clean a single text value by removing non-ASCII characters and normalizing spaces.

        Args:
            valor: Text value to clean
            remove_non_ascii: Whether to remove non-ASCII characters
            normalize_spaces: Whether to normalize all spaces (including special spaces)

        Returns:
            Cleaned text value
        """
        if pd.isna(valor):
            return valor

        valor_str = str(valor).strip()

        if remove_non_ascii:
            valor_str = re.sub(r'[^\x00-\x7F]+', '', valor_str)

        if normalize_spaces:
            valor_str = re.sub(r'\s+', ' ', valor_str)

        return valor_str

    @staticmethod
    def remove_duplicates_advanced(df: pd.DataFrame,
                                   subset: Optional[List[str]] = None,
                                   keep: str = 'first',
                                   report: bool = True) -> pd.DataFrame:
        """
        Remove duplicates with optional reporting.

        Args:
            df: DataFrame to process
            subset: Columns to consider for identifying duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            report: Whether to print duplicate statistics

        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)  # type: ignore
        final_rows = len(df_clean)

        if report:
            duplicates_removed = initial_rows - final_rows
            print(f"Duplicates removed: {duplicates_removed}")
            print(f"Rows before: {initial_rows}, Rows after: {final_rows}")

        return df_clean

    @staticmethod
    def timedelta_series_to_time(series: pd.Series) -> pd.Series:
        """Efficiently convert a pandas Series of timedelta-like values into datetime.time objects.

        Handles pd.Timedelta, datetime.timedelta, numeric seconds, strings, datetime values.
        Preserves NaT/None/NaN as None. For durations longer than 24h, wraps around (modulo 24h).

        Args:
            series (pd.Series): Input Series with timedelta-like or time-like values.

        Returns:
            pd.Series: Series of datetime.time objects (dtype object, None for missing).
        """
        # Convert input to timedelta (handles various input types)
        td = pd.to_timedelta(series, errors='coerce')
        total_seconds = td.dt.total_seconds()
        seconds_mod = total_seconds % (24 * 3600)

        # Initialize output with None
        times = pd.Series([None] * len(series), index=series.index, dtype=object)

        # For valid timedelta values
        mask_valid = seconds_mod.notna()
        if mask_valid.any():
            secs_int = seconds_mod[mask_valid].round().astype(int)
            # Convert seconds -> datetime -> extract time objects
            times.loc[mask_valid] = (
                pd.to_datetime(secs_int, unit='s', origin='unix')
                .dt.time
                .values
            )

        # For entries that didn't convert (e.g., strings or datetimes)
        mask_missing = times.isna()
        if mask_missing.any():
            parsed_dt = pd.to_datetime(series[mask_missing], errors='coerce')
            times.loc[mask_missing] = parsed_dt.dt.time.values

        return times

    @staticmethod
    def fill_missing_by_group(df: pd.DataFrame,
                          group_cols: Union[List[str], str],
                          target_col: str,
                          lower_limit: float = 0,
                          fill_method: Union[str, callable] = 'mean',  # type: ignore
                          min_group_count: int = 1,
                          consider_nans: bool = True,
                          consider_inverse_matches: bool = True,
                          estadio_jogo_col: Optional[str] = None,
                          estadio_casa_col: Optional[str] = None, 
                          as_int: bool = False,
                          inplace: bool = False,
                          verbose: bool = False) -> Union[Tuple[pd.DataFrame, Dict[str, Any]], Dict[str, Any]]:
        """
        Preenche valores em falta ou abaixo de um limite com um agregado por grupo.
        Inclui jogos invertidos (Casa/Fora) e usa zero como fallback final.

        Args:
            df: DataFrame de entrada.
            group_cols: Coluna(s) para agrupar (ex: ['EquipaCasa_cleaned', 'EquipaFora_cleaned']).
            target_col: Coluna onde os valores em falta / baixos serão preenchidos (ex: 'Espetadores').
            lower_limit: Limite mínimo a considerar. Valores <= lower_limit são considerados para preenchimento.
            fill_method: 'mean' (default), 'median', 'mode', 'min', 'max', 'sum' ou uma função callable.
            min_group_count: Número mínimo de observações válidas no grupo para usar o agregado.
            consider_nans: Se True, trata NaNs como valores a preencher.
            consider_inverse_matches: Se True, inclui jogos invertidos (Casa/Fora) ao calcular agregados.
            as_int: Se True, arredonda os valores finais e converte para inteiros.
            inplace: Se True, modifica o DataFrame passado; caso contrário, trabalha sobre cópia.
            verbose: Se True, imprime resumo das operações.

        Returns:
            - Se inplace=False: (df_result, stats)
            - Se inplace=True: stats
        """

        if isinstance(group_cols, str):
            group_cols = [group_cols]

        missing_cols = set(group_cols + [target_col]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"As seguintes colunas não existem no DataFrame: {missing_cols}")

        df_work = df if inplace else df.copy()
        df_work[target_col] = pd.to_numeric(df_work[target_col], errors='coerce')

        # Máscara de valores válidos
        valid_mask = df_work[target_col].notna() & (df_work[target_col] > lower_limit)
        df_valid = df_work.loc[valid_mask, group_cols + [target_col]]

        stats: Dict[str, Any] = {}
        stats['initial_candidates'] = int(((df_work[target_col].isna() & consider_nans) |
                                        (df_work[target_col] <= lower_limit)).sum())

        # Determinar função de agregação
        def _make_agg_func(m):
            if callable(m):
                return m
            m = m.lower()
            if m == 'mean': return lambda s: float(s.mean())
            if m == 'median': return lambda s: float(s.median())
            if m == 'mode': return lambda s: float(s.mode().iat[0]) if not s.mode().empty else np.nan
            if m == 'min': return lambda s: float(s.min())
            if m == 'max': return lambda s: float(s.max())
            if m == 'sum': return lambda s: float(s.sum())
            raise ValueError(f"fill_method desconhecido: {fill_method}")

        agg_func = _make_agg_func(fill_method)

        if df_valid.empty:
            # Nada válido → fallback global = 0
            if verbose:
                print("Sem valores válidos para cálculo. Todos os candidatos serão 0.")
            df_work.loc[(df_work[target_col].isna() & consider_nans) |
                        (df_work[target_col] <= lower_limit), target_col] = 0

            if as_int:
                df_work[target_col] = np.round(df_work[target_col]).astype(int)

            stats.update({'num_candidates': stats['initial_candidates'],
                        'num_filled_from_group': 0,
                        'num_filled_from_fallback': stats['initial_candidates'],
                        'num_filled': stats['initial_candidates'],
                        'num_unfilled_groups': 0,
                        'global_fallback': 0})
            return stats if inplace else (df_work, stats)

        # Agrupa e calcula agregados
        grouped = df_valid.groupby(group_cols)[target_col].agg(lambda s: agg_func(s)).reset_index().rename(
            columns={target_col: f'{target_col}_agg'})
        counts = df_valid.groupby(group_cols).size().reset_index(name='count')
        group_stats = grouped.merge(counts, on=group_cols)
        group_stats = group_stats[group_stats['count'] >= max(1, int(min_group_count))].copy()

        # Duplicar invertidos se aplicável
        if consider_inverse_matches and len(group_cols) >= 2:
            # Só faz sentido duplicar se o estádio do jogo for o mesmo que o estádio de casa
            if estadio_jogo_col and estadio_casa_col and estadio_jogo_col in df_work.columns and estadio_casa_col in df_work.columns:
                # Criar máscara de jogos com estádio igual
                same_stadium_mask = df_work[estadio_jogo_col] == df_work[estadio_casa_col]
                if verbose:
                    print(f"Jogos com estádio igual: {same_stadium_mask.sum()} / {len(df_work)}")
                # Filtrar apenas esses jogos antes de duplicar
                valid_for_swap = group_stats[group_stats[group_cols[0]].isin(
                    df_work.loc[same_stadium_mask, group_cols[0]]
                )]
                swapped = valid_for_swap.copy()
                swapped = swapped.rename(columns={group_cols[0]: group_cols[1], group_cols[1]: group_cols[0]})
                group_stats = pd.concat([group_stats, swapped], ignore_index=True)
            else:
                # Mantém comportamento antigo se não há colunas de estádio
                swapped = group_stats.copy()
                swapped = swapped.rename(columns={group_cols[0]: group_cols[1], group_cols[1]: group_cols[0]})
                group_stats = pd.concat([group_stats, swapped], ignore_index=True)


        # Merge
        df_merged = df_work.merge(group_stats.drop(columns=['count']), on=group_cols, how='left')

        # Máscara de preenchimento
        to_fill_mask = ((df_merged[target_col].isna() & consider_nans) |
                        (df_merged[target_col] <= lower_limit))
        agg_col = f'{target_col}_agg'

        # ✅ garantir que é float antes de preencher
        df_merged[target_col] = df_merged[target_col].astype(float)

        # Preenchimento por grupo
        filled_from_group = to_fill_mask & df_merged[agg_col].notna()
        df_merged.loc[filled_from_group, target_col] = df_merged.loc[filled_from_group, agg_col]

        # Preenchimento por fallback (zero)
        filled_from_fallback = to_fill_mask & df_merged[agg_col].isna()
        df_merged.loc[filled_from_fallback, target_col] = 0

        # Se solicitado, arredonda e converte para inteiro
        if as_int:
            df_merged[target_col] = np.round(df_merged[target_col].fillna(0)).astype(int)

        # Estatísticas
        stats['num_candidates'] = int(to_fill_mask.sum())
        stats['num_filled_from_group'] = int(filled_from_group.sum())
        stats['num_filled_from_fallback'] = int(filled_from_fallback.sum())
        stats['num_filled'] = stats['num_filled_from_group'] + stats['num_filled_from_fallback']
        stats['num_unfilled_groups'] = 0
        stats['global_fallback'] = 0

        if inplace:
            df[target_col] = df_merged[target_col]
            if verbose:
                print("fill_missing_by_group summary (inplace):")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            return stats
        else:
            df_result = df_merged.drop(columns=[agg_col])
            if verbose:
                print("fill_missing_by_group summary:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            return df_result, stats

    @staticmethod
    def replace_nan_values(df: pd.DataFrame, values_to_replace: Optional[List] = None) -> pd.DataFrame:
        """
        Replaces common null-like values (None, 'None', 'NaN', '<NA>', 'Missing value', etc.)
        with pandas.NA across the entire DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to clean.
            values_to_replace (Optional[List]): Additional values to replace.
                                                Defaults to a common list of null-like strings.

        Returns:
            pd.DataFrame: A cleaned copy of the DataFrame with standardized missing values.
        """
        # Default replacements: covers text and Python nulls
        default_nulls = [
            None, 'None', 'none', 'NaN', 'nan', 'NAN', 
            '<NA>', 'NA', 'N/A', 'n/a', 'Null', 'NULL',
            'Missing', 'Missing value', 'missing', 'missing value', ''
        ]
        
        # Merge custom and default values
        if values_to_replace is not None:
            null_values = list(set(default_nulls + values_to_replace))
        else:
            null_values = default_nulls

        # Perform replacement
        df = df.replace(null_values, pd.NA)

        return df

    @staticmethod
    def parse_data_portuguesa(data_str: str) -> pd.Timestamp:
        """
        Converte strings de data/hora em português como:
            - 'sab. 18 ago 2012 17h15'
            - 'sab. 02 ago 2008'
        para formato ISO UTC ('YYYY-MM-DD HH:MM:SS+00:00')
        """
        meses_pt = {
            # português
            'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
            'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12,
            # inglês
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        # Limpeza
        s = str(data_str).strip().lower().replace('.', '').replace('h', ':')
        partes = s.split()

        if len(partes) < 4:
            raise ValueError(f"Formato inesperado: {data_str}")

        # Exemplo: ['sab', '02', 'ago', '2008', '17:15']
        dia = partes[1]
        mes_abrev = partes[2]
        ano = partes[3]
        hora = partes[4] if len(partes) >= 5 else "00:00"  # <–– hora padrão

        mes = meses_pt.get(mes_abrev)
        if mes is None:
            raise ValueError(f"Mês desconhecido: {mes_abrev}")

        # Construir datetime
        try:
            data_hora = datetime.strptime(f"{ano}-{mes:02d}-{dia} {hora}", "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError(f"Formato inválido: {data_str}")

        return pd.Timestamp(data_hora, tz="UTC")

    @staticmethod
    def camel_to_snake(name: str) -> str:
        """Convert CamelCase or PascalCase to snake_case."""
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

    @staticmethod
    def group_dynamic(df, keys, aggregations, dropna=False, verbose=True):
        """
        Dynamically groups a DataFrame based on provided key columns and aggregation rules.
        Automatically includes non-aggregated columns using 'first' by default.
        """

        missing_keys = [c for c in keys if c not in df.columns]
        if missing_keys:
            raise KeyError(f"Grouping columns not found in DataFrame: {missing_keys}")

        aggregation_columns = [v[0] for v in aggregations.values() if isinstance(v, tuple)]
        missing_aggregations = [c for c in aggregation_columns if c not in df.columns]
        if missing_aggregations:
            raise KeyError(f"Aggregation columns not found in DataFrame: {missing_aggregations}")

        duplicates = df.duplicated(subset=keys, keep=False)
        num_duplicates = duplicates.sum()

        if num_duplicates == 0:
            if verbose:
                print("No duplicates found. Returning original DataFrame.")
            return df.copy()

        if verbose:
            print(f"{num_duplicates} duplicates found. Applying grouping...")

        # Include any non-aggregated, non-key columns using 'first'
        extra_cols = [c for c in df.columns if c not in keys and c not in aggregation_columns]
        for col in extra_cols:
            if col not in aggregations:
                aggregations[col] = (col, 'first')

        df_grouped = (
            df.groupby(keys, dropna=dropna)
            .agg(**aggregations)
            .reset_index()
        )

        if verbose:
            print("Grouping completed successfully.")
        return df_grouped

class DataAnalysis:
    """
    Collection of functions for data analysis and exploration.
    """

    @staticmethod
    def quick_summary(df: pd.DataFrame, show_dtypes: bool = True, show_plots: bool = True) -> None:
        """
        Generate a quick summary of the DataFrame with optional visualizations.

        Args:
            df: DataFrame to analyze
            show_dtypes: Whether to show data types
            show_plots: Whether to show exploratory plots
        """
        print("=" * 50)
        print("QUICK DATA SUMMARY")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        if show_dtypes:
            print("\nData Types:")
            print(df.dtypes.value_counts())

        print("\nMissing Values:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_summary = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        print(missing_summary[missing_summary['Missing Count'] > 0])

        print("\nNumeric Columns Summary:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())

        # Gráficos exploratórios
        if show_plots and len(df) > 0:
            print("\nGenerating exploratory plots...")
            
            # 1. Heatmap de valores missing (se existirem)
            if missing.sum() > 0:
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
                plt.title('Missing Values Heatmap')
                plt.tight_layout()
                plt.show()
            
            # 2. Distribuição das variáveis numéricas (excluindo timedelta)
            # Filtrar apenas tipos numéricos puros (int, float)
            numeric_cols_plot = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns
            
            if len(numeric_cols_plot) > 0:
                n_cols = min(len(numeric_cols_plot), 4)  # Máximo 4 colunas por linha
                n_rows = (len(numeric_cols_plot) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                axes = axes.flatten() if len(numeric_cols_plot) > 1 else [axes]
                
                for idx, col in enumerate(numeric_cols_plot):
                    axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    axes[idx].set_title(f'Distribution: {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
                
                # Remove eixos vazios
                for idx in range(len(numeric_cols_plot), len(axes)):
                    fig.delaxes(axes[idx])
                
                plt.tight_layout()
                plt.show()
            
            # 3. Boxplots para detectar outliers
            if len(numeric_cols_plot) > 0:
                n_cols_box = min(len(numeric_cols_plot), 3)
                n_rows_box = (len(numeric_cols_plot) + n_cols_box - 1) // n_cols_box
                
                fig, axes = plt.subplots(n_rows_box, n_cols_box, figsize=(6*n_cols_box, 4*n_rows_box))
                axes = axes.flatten() if len(numeric_cols_plot) > 1 else [axes]
                
                for idx, col in enumerate(numeric_cols_plot):
                    axes[idx].boxplot(df[col].dropna(), vert=True)
                    axes[idx].set_title(f'Boxplot: {col}')
                    axes[idx].set_ylabel(col)
                
                for idx in range(len(numeric_cols_plot), len(axes)):
                    fig.delaxes(axes[idx])
                
                plt.tight_layout()
                plt.show()
            
            # 4. Matriz de correlação (se houver múltiplas variáveis numéricas)
            if len(numeric_cols_plot) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = df[numeric_cols_plot].corr()
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                        center=0, square=True, linewidths=1)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                plt.show()
            
            # 5. Tratamento especial para colunas timedelta/time
            timedelta_cols = df.select_dtypes(include=['timedelta64']).columns
            
            if len(timedelta_cols) > 0:
                for col in timedelta_cols:
                    # Verificar se é uma hora do dia (valores entre 0-24h)
                    max_hours = df[col].dt.total_seconds().max() / 3600
                    
                    if max_hours <= 24:  # É uma hora do dia
                        print(f"\n'{col}' appears to be time of day")
                        
                        fig = plt.figure(figsize=(14, 5))
                        
                        # Converter para horas do dia
                        hours = df[col].dt.total_seconds() / 3600
                        hours_clean = hours.dropna()
                        
                        # Gráfico 1: Histograma por hora
                        #ax1 = fig.add_subplot(121)
                        #ax1.hist(hours_clean, bins=24, range=(0, 24), 
                        #        edgecolor='black', alpha=0.7)
                        #ax1.set_title(f'Distribution of {col}')
                        #ax1.set_xlabel('Hour of Day')
                        #ax1.set_ylabel('Frequency')
                        #ax1.set_xticks(range(0, 25, 2))
                        #ax1.grid(axis='y', alpha=0.3)
                        
                        # Gráfico 2: Gráfico polar (relógio)
                        ax2 = fig.add_subplot(122, projection='polar')
                        
                        # Contar jogos por hora
                        hour_counts = hours_clean.value_counts().sort_index()
                        theta = np.array(hour_counts.index * 2 * np.pi / 24)
                        radii = np.array(hour_counts.values)
                        width = 2 * np.pi / 24
                        
                        ax2.bar(theta, radii, width=width, alpha=0.7)
                        ax2.set_theta_zero_location('N')  # type: ignore
                        ax2.set_theta_direction(-1)  # type: ignore
                        ax2.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
                        ax2.set_xticklabels([f'{h:02d}h' for h in range(24)])
                        ax2.set_title(f'{col} - Clock View')
                        
                        plt.tight_layout()
                        plt.show()
                        
                    else:  # É uma duração
                        print(f"\n'{col}' appears to be a duration")
                        
                        plt.figure(figsize=(10, 5))
                        
                        # Converter para minutos ou horas dependendo da escala
                        if max_hours < 2:
                            values = df[col].dt.total_seconds() / 60
                            unit = 'minutes'
                        else:
                            values = df[col].dt.total_seconds() / 3600
                            unit = 'hours'
                        
                        plt.hist(values.dropna(), bins=30, edgecolor='black', alpha=0.7)
                        plt.title(f'Distribution: {col}')
                        plt.xlabel(f'{col} ({unit})')
                        plt.ylabel('Frequency')
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        plt.show()
            
            # 6. Contagem de categorias (variáveis categóricas com poucos valores únicos)
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            categorical_cols = [col for col in categorical_cols 
                            if df[col].nunique() <= 10 and col.lower() != 'hora']
            
            if len(categorical_cols) > 0:
                n_cols_cat = min(len(categorical_cols), 2)
                n_rows_cat = (len(categorical_cols) + n_cols_cat - 1) // n_cols_cat
                
                fig, axes = plt.subplots(n_rows_cat, n_cols_cat, figsize=(8*n_cols_cat, 5*n_rows_cat))
                axes = axes.flatten() if len(categorical_cols) > 1 else [axes]
                
                for idx, col in enumerate(categorical_cols):
                    value_counts = df[col].value_counts()
                    axes[idx].bar(range(len(value_counts)), value_counts.values)
                    axes[idx].set_xticks(range(len(value_counts)))
                    axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
                    axes[idx].set_title(f'Distribution: {col}')
                    axes[idx].set_ylabel('Count')
                
                for idx in range(len(categorical_cols), len(axes)):
                    fig.delaxes(axes[idx])
                
                plt.tight_layout()
                plt.show()

    @staticmethod
    def find_outliers_iqr(df: pd.DataFrame, column: str,
                          multiplier: float = 1.5) -> pd.DataFrame:
        """
        Find outliers using IQR method.

        Args:
            df: DataFrame to analyze
            column: Column to check for outliers
            multiplier: IQR multiplier (default 1.5)

        Returns:
            DataFrame with outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        print(f"Outliers found: {len(outliers)}")
        print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")

        return outliers

    @staticmethod
    def display_column_types_table(df: pd.DataFrame) -> None:
        """
        Exibe uma tabela com o nome das features na primeira linha
        e o tipo de dados na segunda.

        Args:
            df (pd.DataFrame): DataFrame a analisar.

        Raises:
            ValueError: Se o argumento não for um DataFrame ou se estiver vazio.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("O objeto fornecido não é um pandas DataFrame.")
        if df.empty:
            raise ValueError("O DataFrame fornecido está vazio.")
        if df.columns.empty:
            raise ValueError("O DataFrame não contém colunas.")

        # Cria uma linha com os tipos
        type_row = pd.DataFrame([df.dtypes.astype(str).tolist()], columns=df.columns)

        # Exibe as colunas e os tipos (tipos na 2ª linha)
        try:
            display(pd.concat([pd.DataFrame([df.columns], columns=df.columns),
                               type_row], ignore_index=True))
        except NameError:
            # Caso o display() (IPython) não esteja disponível, usa print() como fallback
            print(pd.concat([pd.DataFrame([df.columns], columns=df.columns),
                             type_row], ignore_index=True))

class DataValidation:
    """
    Collection of functions for data validation and quality checks.
    """

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format.

        Args:
            email: Email string to validate

        Returns:
            bool: True if valid email format
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_phone(phone: str, country_code: str = 'PT') -> bool:
        """
        Validate phone number format (basic validation).

        Args:
            phone: Phone number string
            country_code: Country code for validation rules

        Returns:
            bool: True if valid format
        """
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)

        if country_code == 'PT':
            # Portuguese phone numbers: 9 digits starting with 9, or with +351
            if len(digits_only) == 9 and digits_only.startswith('9'):
                return True
            elif len(digits_only) == 12 and digits_only.startswith('351'):
                return True

        return False

    @staticmethod
    def check_data_quality(df: pd.DataFrame,
                           required_columns: List[str] = None) -> Dict[str, Any]:  # type: ignore
        """
        Comprehensive data quality check.

        Args:
            df: DataFrame to check
            required_columns: List of columns that must exist

        Returns:
            Dict with quality metrics
        """
        quality_report = {}

        # Basic info
        quality_report['total_rows'] = len(df)
        quality_report['total_columns'] = len(df.columns)

        # Missing values
        quality_report['missing_values'] = df.isnull().sum().to_dict()
        quality_report['missing_percentage'] = (df.isnull().sum() / len(df) * 100).to_dict()

        # Duplicate rows
        quality_report['duplicate_rows'] = df.duplicated().sum()

        # Required columns check
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            quality_report['missing_required_columns'] = list(missing_cols)

        # Data types
        quality_report['data_types'] = df.dtypes.to_dict()

        return quality_report

class FileOperations:
    """
    Collection of functions for file operations and I/O.
    """

    @staticmethod
    def safe_read_csv(file_path: str, logger: logging.Logger, **kwargs) -> pd.DataFrame:
        """
        Safely read CSV file with error handling.

        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame or None if error
        """
        try:
            # Verificar se o ficheiro existe
            if not os.path.exists(file_path):
                logger.info(f"Ficheiro não encontrado: {file_path}")
                raise FileNotFoundError(f"O ficheiro {file_path} não foi encontrado.")

            # Verificar se é um ficheiro (não um diretório)
            if not os.path.isfile(file_path):
                logger.info(f"O caminho especificado não é um ficheiro: {file_path}")
                raise ValueError(f"O caminho {file_path} não aponta para um ficheiro válido.")
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded {file_path}")
            return df
        except Exception as e:
            logger.info(f"Error reading {file_path}: {str(e)}")
            return None  # type: ignore

    @staticmethod
    def safe_to_csv(df: pd.DataFrame, file_path: str, **kwargs) -> bool:
        """
        Safely save DataFrame to CSV with error handling.

        Args:
            df: DataFrame to save
            file_path: Path where to save
            **kwargs: Additional arguments for to_csv

        Returns:
            bool: True if successful
        """
        try:
            df.to_csv(file_path, **kwargs)
            print(f"Successfully saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving to {file_path}: {str(e)}")
            return False

    @staticmethod
    def safe_read_text_file(file_path: str, logger: logging.Logger) -> Optional[str]:
        """
        Safely reads the content of a text file with error handling.
        
        Args:
            file_path: The path to the text file.
            logger: The logger instance to log messages.
            
        Returns:
            The content of the file as a string, or None if an error occurs.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                logger.info(f"Successfully read content from {file_path}")
                return content
        except FileNotFoundError:
            logger.error(f"Error: The file was not found at {file_path}.")
            return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    @staticmethod
    def safe_read_csv_with_validation(
        file_path: str,
        logger: logging.Logger,
        required_columns: List[str],
        unique_columns: Optional[List[str]] = None,
        raise_on_empty: bool = True,
        raise_on_missing_columns: bool = True,
        raise_on_duplicate_values: bool = True,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Lê um ficheiro CSV com validação de colunas obrigatórias e valores únicos.
        
        Args:
            file_path: Caminho para o ficheiro CSV
            logger: Instância do logger para registar mensagens
            required_columns: Lista de colunas que devem existir no DataFrame
            unique_columns: Lista de colunas (ou combinações) que devem ter valores únicos.
                        A validação só ocorre para registos com 'is_current'=True (SCD).
                        Pode ser uma lista de strings para colunas individuais,
                        ou uma lista de listas para validar unicidade composta.
                        Ex: ['id'] ou [['club_id', 'stadium_id']]
            raise_on_empty: Se True, lança exceção quando o DataFrame está vazio
            raise_on_missing_columns: Se True, lança exceção quando faltam colunas obrigatórias
            raise_on_duplicate_values: Se True, lança exceção quando encontra valores duplicados
            **kwargs: Argumentos adicionais para pd.read_csv (sep, decimal, etc.)
        
        Returns:
            DataFrame com os dados ou None em caso de erro (se raise_on_* = False)
        
        Raises:
            FileNotFoundError: Se o ficheiro não existir
            ValueError: Se o DataFrame estiver vazio, faltar colunas ou houver duplicados
        
        Example:
            >>> required_cols = ['club_id', 'stadium_id', 'is_current']
            >>> unique_cols = ['stadium_id']  # stadium_id deve ser único nos registos atuais
            >>> df = FileOperations.safe_read_csv_with_validation(
            ...     'stadiums.csv',
            ...     logger,
            ...     required_cols,
            ...     unique_columns=unique_cols,
            ...     sep=';',
            ...     decimal='.'
            ... )
            
            >>> # Validar unicidade composta apenas em registos atuais
            >>> unique_cols = [['club_id', 'stadium_id']]
            >>> df = FileOperations.safe_read_csv_with_validation(
            ...     'club_stadiums.csv',
            ...     logger,
            ...     required_cols,
            ...     unique_columns=unique_cols,
            ...     sep=';'
            ... )
        """
        logger.info(f"Tentando carregar ficheiro: {file_path}")
        
        # Tenta ler o ficheiro usando safe_read_csv
        df = FileOperations.safe_read_csv(file_path, logger, **kwargs)
        
        # Se safe_read_csv retornou None (erro na leitura)
        if df is None:
            logger.error(f"Erro ao carregar o ficheiro: {file_path}")
            if raise_on_empty:
                raise FileNotFoundError(f"Não foi possível carregar o ficheiro: {file_path}")
            return None
        
        # Verificar se o DataFrame está vazio
        if df.empty:
            logger.warning(f"O ficheiro '{file_path}' foi carregado mas está vazio.")
            if raise_on_empty:
                raise ValueError(f"O ficheiro '{file_path}' está vazio.")
            return None
        
        # Verificar se as colunas obrigatórias existem
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Colunas obrigatórias em falta no ficheiro '{file_path}': {missing_columns}")
            if raise_on_missing_columns:
                raise ValueError(
                    f"O ficheiro '{file_path}' não contém as colunas necessárias: {missing_columns}"
                )
            return None
        
        # Verificar valores únicos se especificado
        if unique_columns:
            # Verificar se existe a coluna 'is_current' para aplicar a validação SCD
            if 'is_current' not in df.columns:
                logger.warning(
                    f"Coluna 'is_current' não encontrada no ficheiro '{file_path}'. "
                    f"A validação de unicidade não será aplicada (apenas para tabelas SCD)."
                )
            else:
                # Filtrar apenas os registos atuais (is_current = True)
                df_current = df[df['is_current'] == True]
                
                if df_current.empty:
                    logger.warning(
                        f"Nenhum registo com 'is_current'=True encontrado no ficheiro '{file_path}'. "
                        f"Validação de unicidade não aplicada."
                    )
                else:
                    logger.info(
                        f"Validando unicidade em {len(df_current)} registos atuais "
                        f"(is_current=True) de um total de {len(df)} registos."
                    )
                    
                    for col_spec in unique_columns:
                        # Se for uma lista de colunas (unicidade composta)
                        if isinstance(col_spec, list):
                            cols = col_spec
                            col_name = " + ".join(cols)
                            
                            # Verificar se todas as colunas existem
                            missing = [c for c in cols if c not in df.columns]
                            if missing:
                                logger.error(f"Colunas para validação de unicidade não encontradas: {missing}")
                                if raise_on_duplicate_values:
                                    raise ValueError(
                                        f"Colunas necessárias para validação de unicidade não existem: {missing}"
                                    )
                                return None
                            
                            # Verificar duplicados na combinação de colunas (apenas registos atuais)
                            duplicates = df_current[df_current.duplicated(subset=cols, keep=False)]
                            
                        # Se for uma coluna individual
                        else:
                            cols = [col_spec]
                            col_name = col_spec
                            
                            # Verificar se a coluna existe
                            if col_spec not in df.columns:
                                logger.error(f"Coluna para validação de unicidade não encontrada: {col_spec}")
                                if raise_on_duplicate_values:
                                    raise ValueError(
                                        f"Coluna necessária para validação de unicidade não existe: {col_spec}"
                                    )
                                return None
                            
                            # Verificar duplicados (apenas registos atuais)
                            duplicates = df_current[df_current.duplicated(subset=cols, keep=False)]
                        
                        # Se encontrou duplicados
                        if not duplicates.empty:
                            num_duplicates = duplicates.shape[0]
                            duplicate_values = duplicates[cols].drop_duplicates()
                            
                            logger.error(
                                f"Encontrados {num_duplicates} registos ATUAIS (is_current=True) com valores duplicados "
                                f"na(s) coluna(s) '{col_name}' no ficheiro '{file_path}'"
                            )
                            logger.error(f"Valores duplicados:\n{duplicate_values.to_string()}")
                            
                            if raise_on_duplicate_values:
                                raise ValueError(
                                    f"O ficheiro '{file_path}' contém valores duplicados na(s) coluna(s) "
                                    f"'{col_name}' nos registos atuais (is_current=True). "
                                    f"Encontrados {num_duplicates} registos duplicados."
                                )
                            return None
                        else:
                            logger.info(
                                f"Validação de unicidade OK para coluna(s) '{col_name}' "
                                f"nos registos atuais (is_current=True)"
                            )
        
        logger.info(f"Ficheiro '{file_path}' carregado com sucesso. Shape: {df.shape}")
        logger.info("Todas as colunas necessárias estão presentes.")
        
        return df

class Logger:
    """Logger utility class with configurable log directory."""
    
    @staticmethod
    def setup_logger(name: str, log_file: Optional[str] = None,
                     level: int = logging.INFO,
                     log_dir: str = 'logs',
                     max_age_days: int = 30) -> logging.Logger:
        """
        Cria e configura um logger. Se o ficheiro de log for mais antigo que 
        'max_age_days', o seu conteúdo é truncado (apagado) antes de escrever novos dados.

        Args:
            name: Nome do Logger.
            log_file: Caminho para o ficheiro de log (opcional).
            level: Nível de Log (default: INFO).
            log_dir: Diretório base para os ficheiros de log (default: 'logs').
            max_age_days: Idade máxima do ficheiro em dias. Se o ficheiro for mais antigo, 
                          o seu conteúdo é apagado. Se for <= 0, a truncagem é ignorada. 
                          (Default: 30)

        Returns:
            logging.Logger: Instância do logger configurada.
        """
        
        if log_file is None:
            log_file = os.path.join(log_dir, f'{name}.log')
        
        log_path = os.path.dirname(log_file)
        if log_path and not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
            
        # Determinar o modo de abertura do ficheiro: 'w' (write/truncar) ou 'a' (append/anexar)
        file_mode = 'a'
        
        if os.path.exists(log_file) and max_age_days > 0:
            # Tempo de corte: qualquer coisa anterior a este carimbo de data/hora é considerada "antiga"
            cutoff_time = time.time() - (max_age_days * 86400) # Segundos desde a Época
            
            # Hora da última modificação do ficheiro
            file_modified_time = os.path.getmtime(log_file)
            
            if file_modified_time < cutoff_time:
                # O ficheiro é mais antigo que a idade máxima: usar o modo 'w' para o truncar
                file_mode = 'w'
                # Usa um print como fallback para informar que a limpeza vai ocorrer
                print(f"INFO - O log '{log_file}' tem mais de {max_age_days} dias. O seu conteúdo será apagado.")
            else:
                # O ficheiro é recente: usar o modo 'a' para anexar
                print(f"INFO - O log '{log_file}' é recente. O novo conteúdo será anexado.")


        # 4. Configurar o logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        
        # O handler existente deve ser removido para aplicar o novo modo ('w' ou 'a')
        # Apenas remove handlers FileHandler que correspondam ao ficheiro atual
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file):
                logger.removeHandler(handler)
        

        # 5. Adicionar o novo FileHandler com o modo determinado
        handler = logging.FileHandler(log_file, mode=file_mode)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    @staticmethod
    def setup_console_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """
        Create and setup a logger with console handler.

        Args:
            name: Logger name
            level: Logging level (default: INFO)

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @staticmethod
    def setup_dual_logger(name: str, log_file: Optional[str] = None,
                          level: int = logging.INFO,
                          log_dir: str = 'logs') -> logging.Logger:
        """
        Create and setup a logger with both file and console handlers.

        Args:
            name: Logger name
            log_file: Path to log file (optional, will use '{log_dir}/{name}.log' if not provided)
            level: Logging level (default: INFO)
            log_dir: Base directory for log files (default: 'logs')

        Returns:
            logging.Logger: Configured logger instance

        Example:
            >>> logger = Logger.setup_dual_logger('my_app', log_dir='/var/log/myapp')
            >>> logger.info('This goes to both file and console')
        """
        if log_file is None:
            log_file = os.path.join(log_dir, f'{name}.log')

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        if not logger.handlers:
            # Ensure logs directory exists
            log_path = os.path.dirname(log_file)
            if log_path and not os.path.exists(log_path):
                os.makedirs(log_path, exist_ok=True)

            # File handler
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    @staticmethod
    def log_dataframe_info(logger: logging.Logger, df: pd.DataFrame,
                           df_name: str = "DataFrame") -> None:
        """
        Log basic information about a DataFrame.

        Args:
            logger: Logger instance
            df: DataFrame to log info about
            df_name: Name to use in log messages
        """
        logger.info(f"{df_name} shape: {df.shape}")
        logger.info(f"{df_name} memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"{df_name} has {missing_count} missing values")
        else:
            logger.info(f"{df_name} has no missing values")

    @staticmethod
    def log_function_execution(logger: logging.Logger, func_name: str,
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None,
                              success: bool = True,
                              error: Optional[str] = None) -> None:
        """
        Log function execution details.

        Args:
            logger: Logger instance
            func_name: Name of the function
            start_time: Start time (timestamp)
            end_time: End time (timestamp)
            success: Whether function executed successfully
            error: Error message if function failed
        """
        if start_time and end_time:
            execution_time = end_time - start_time
            logger.info(f"{func_name} executed in {execution_time:.2f} seconds")

        if success:
            logger.info(f"{func_name} completed successfully")
        else:
            logger.error(f"{func_name} failed: {error}")
            
class DatabaseManager:

    def __init__(self, DB_CONFIG, logger, joint_query_file: Optional[str] = None, sql_base_path: Optional[str] = None):
        self.logger = logger
        self.connection = None
        self.cursor = None
        self.DB_CONFIG = DB_CONFIG
        self.joint_query_file = joint_query_file
        self.sql_base_path = sql_base_path

        # --- Definição das queries de criação das tabelas ---
        # Query para criar a tabela 'Audiencias'
        self.CREATE_AUDIENCIAS_TABLE_QUERY = """
                CREATE TABLE `{database_name}`.`Audiencias` (
                    `Epoca` text,
                    `Codigo` text,
                    `Competicao` text,
                    `Date` text,
                    `Round` text,
                    `Station` text,
                    `StartTime` text,
                    `Duration` text,
                    `Title` text,
                    `EquipaCasa` text,
                    `EquipaFora` text,
                    `Rat` text,
                    `Shr` double DEFAULT NULL,
                    `Rch` double DEFAULT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """

        # Query para criar a tabela 'Espectadores'
        self.CREATE_ESPECTADORES_TABLE_QUERY = """
                CREATE TABLE `{database_name}`.`Espectadores` (
                    `Epoca` text,
                    `Jornada` bigint DEFAULT NULL,
                    `Data` text,
                    `DiadaSemana` text,
                    `Hora` text,
                    `EquipaCasa` text,
                    `EquipaFora` text,
                    `Jogo` text,
                    `Espetadores` text,
                    `Competicao` text,
                    `TaxaOcupacao` text
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
            """
        # Dicionário com as configurações das tabelas
        self.TABLE_CONFIGURATIONS = {
            "Audiencias": self.CREATE_AUDIENCIAS_TABLE_QUERY,
            "Espectadores": self.CREATE_ESPECTADORES_TABLE_QUERY,
        }

         # Atribui a joint_query a partir do ficheiro, se fornecido
        if joint_query_file:
            # Resolve o caminho completo do ficheiro SQL
            if sql_base_path:
                full_path = os.path.join(sql_base_path, joint_query_file)
            else:
                full_path = joint_query_file

            # Tenta ler a query do ficheiro
            self.joint_query = FileOperations.safe_read_text_file(full_path, self.logger)
            if not self.joint_query:
                self.logger.error("Falha ao carregar a joint_query do ficheiro. Usando a string padrão.")
                self.joint_query = """
                SELECT
                    ltrim(rtrim(E.Epoca)) as Epoca,
                    ltrim(rtrim(E.Competicao)) as Competicao,
                    CAST(REGEXP_REPLACE(E.Jornada, '[^0-9]', '') AS UNSIGNED) AS Jornada,
                    CASE
                        WHEN E.Data LIKE '%/%/%' THEN STR_TO_DATE(E.Data, '%m/%d/%Y')
                        ELSE STR_TO_DATE(E.Data, '%d-%m-%Y')
                    END AS Data,
                    CASE
                        WHEN E.Hora = '0:00' OR E.Hora = '00:00' THEN STR_TO_DATE(A.StartTime, '%H:%i:%s')
                        ELSE STR_TO_DATE(E.Hora, '%H:%i')
                    END AS Hora,
                    ltrim(rtrim(E.EquipaCasa)) as EquipaCasa,
                    ltrim(rtrim(E.EquipaFora)) as EquipaFora,
                    CAST(REGEXP_REPLACE(E.Espetadores, '[^0-9]', '') AS UNSIGNED) AS Espetadores,
                    CAST(REPLACE(REPLACE(ltrim(rtrim(E.TaxaOcupacao)), '%', ''), ',', '.') AS DECIMAL(10, 2)) AS TaxaOcupacao,
                    ltrim(rtrim(A.Codigo)) as Codigo,
                    ltrim(rtrim(A.Station)) as Station,
                    CAST(REGEXP_REPLACE(ltrim(rtrim(A.Rat)), '[^0-9]', '') AS DECIMAL(10, 2)) AS Rat,
                    ltrim(rtrim(A.Shr)) as Shr,
                    ltrim(rtrim(A.Rch)) as Rch
                FROM
                    Espectadores AS E
                LEFT JOIN
                    Audiencias AS A ON
                    UPPER(TRIM(REGEXP_REPLACE(E.Epoca, '[[:space:]]+', ' '))) = UPPER(TRIM(REGEXP_REPLACE(A.Epoca, '[[:space:]]+', ' ')))
                    AND UPPER(TRIM(REGEXP_REPLACE(E.Competicao, '[[:space:]]+', ' '))) = UPPER(TRIM(REGEXP_REPLACE(A.Competicao, '[[:space:]]+', ' ')))
                    AND CAST(REGEXP_REPLACE(E.Jornada, '[^0-9]', '') AS UNSIGNED) = CAST(REGEXP_REPLACE(A.Round, '[^0-9]', '') AS UNSIGNED)
                    AND UPPER(TRIM(REGEXP_REPLACE(E.EquipaCasa, '[[:space:]]+', ' '))) = UPPER(TRIM(REGEXP_REPLACE(A.EquipaCasa, '[[:space:]]+', ' ')))
                    AND UPPER(TRIM(REGEXP_REPLACE(E.EquipaFora, '[[:space:]]+', ' '))) = UPPER(TRIM(REGEXP_REPLACE(A.EquipaFora, '[[:space:]]+', ' ')))
                WHERE E.Espetadores IS NOT NULL AND E.Competicao NOT LIKE '%Allianz%'
                ORDER BY E.Epoca, E.Jornada, E.Data;"""
        else:
            self.joint_query = """
            SELECT
                ltrim(rtrim(E.Epoca)) as Epoca,
                ltrim(rtrim(E.Competicao)) as Competicao,
                CAST(REGEXP_REPLACE(E.Jornada, '[^0-9]', '') AS UNSIGNED) AS Jornada,
                CASE
                    WHEN E.Data LIKE '%/%/%' THEN STR_TO_DATE(E.Data, '%m/%d/%Y')
                    ELSE STR_TO_DATE(E.Data, '%d-%m-%Y')
                END AS Data,
                CASE
                    WHEN E.Hora = '0:00' OR E.Hora = '00:00' THEN STR_TO_DATE(A.StartTime, '%H:%i:%s')
                    ELSE STR_TO_DATE(E.Hora, '%H:%i')
                END AS Hora,
                ltrim(rtrim(E.EquipaCasa)) as EquipaCasa,
                ltrim(rtrim(E.EquipaFora)) as EquipaFora,
                CAST(REGEXP_REPLACE(E.Espetadores, '[^0-9]', '') AS UNSIGNED) AS Espetadores,
                CAST(REPLACE(REPLACE(ltrim(rtrim(E.TaxaOcupacao)), '%', ''), ',', '.') AS DECIMAL(10, 2)) AS TaxaOcupacao,
                ltrim(rtrim(A.Codigo)) as Codigo,
                ltrim(rtrim(A.Station)) as Station,
                CAST(REGEXP_REPLACE(ltrim(rtrim(A.Rat)), '[^0-9]', '') AS DECIMAL(10, 2)) AS Rat,
                ltrim(rtrim(A.Shr)) as Shr,
                ltrim(rtrim(A.Rch)) as Rch
            FROM
                Espectadores AS E
            LEFT JOIN
                Audiencias AS A ON
                UPPER(TRIM(REGEXP_REPLACE(E.Epoca, '[[:space:]]+', ' '))) = UPPER(TRIM(REGEXP_REPLACE(A.Epoca, '[[:space:]]+', ' ')))
                AND UPPER(TRIM(REGEXP_REPLACE(E.Competicao, '[[:space:]]+', ' '))) = UPPER(TRIM(REGEXP_REPLACE(A.Competicao, '[[:space:]]+', ' ')))
                AND CAST(REGEXP_REPLACE(E.Jornada, '[^0-9]', '') AS UNSIGNED) = CAST(REGEXP_REPLACE(A.Round, '[^0-9]', '') AS UNSIGNED)
                AND UPPER(TRIM(REGEXP_REPLACE(E.EquipaCasa, '[[:space:]]+', ' '))) = UPPER(TRIM(REGEXP_REPLACE(A.EquipaCasa, '[[:space:]]+', ' ')))
                AND UPPER(TRIM(REGEXP_REPLACE(E.EquipaFora, '[[:space:]]+', ' '))) = UPPER(TRIM(REGEXP_REPLACE(A.EquipaFora, '[[:space:]]+', ' ')))
            WHERE E.Espetadores IS NOT NULL AND E.Competicao NOT LIKE '%Allianz%'
            ORDER BY E.Epoca, E.Jornada, E.Data;"""


    def connect(self):
        """
        Estabelece uma conexão com a base de dados e inicializa o cursor.
        Retorna True em caso de sucesso, False em caso de falha.
        """
        try:
            self.connection = mysql.connector.connect(**self.DB_CONFIG)
            if self.connection.is_connected():
                self.cursor = self.connection.cursor(cursor_class=MySQLCursorDict)
                self.logger.info("Conexão à base de dados estabelecida com sucesso")
                return True
            else:
                self.logger.error("Falha ao conectar à base de dados: Conexão não estabelecida após chamada.")
                return False
        except Error as e:
            self.logger.error(f"Erro ao conectar à base de dados: {e}")
            self.connection = None
            self.cursor = None
            return False

    def disconnect(self):
        """
        Fecha o cursor e a conexão com a base de dados, se estiverem abertos.
        """
        if self.cursor:
            self.cursor.close()
            self.cursor = None

        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.connection = None
            self.logger.info("Conexão à base de dados fechada")
        elif self.connection is None:
            self.logger.info("Nenhuma conexão ativa para fechar.")
        else:
            self.logger.info("Conexão já fechada ou inválida.")

    def execute_query(self, query, params=None, commit=False):
        """
        Executa uma query SQL. Tenta reconectar se a conexão não estiver ativa.
        Retorna os resultados para SELECTs ou True/False para operações com commit.
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.logger.warning("Conexão à base de dados não ativa. Tentando reconectar...")
                if not self.connect():
                    self.logger.error("Falha ao reconectar à base de dados para executar a query.")
                    return None

            if self.cursor is None:
                self.logger.error("Cursor não inicializado após a conexão. Não é possível executar a query.")
                return None

            self.cursor.execute(query, params or ())

            if commit:
                if self.connection:
                    self.connection.commit()
                    self.logger.info("Transação commitada com sucesso.")
                    return True
                else:
                    self.logger.error("Conexão inválida para realizar o commit.")
                    return False
            else:
                result = self.cursor.fetchall()
                return result
        except Error as e:
            self.logger.error(f"Erro ao executar query: {e}")
            self.logger.error(f"Query: {query}")
            if params:
                self.logger.error(f"Parâmetros: {params}")

            if commit and self.connection and self.connection.is_connected():
                try:
                    self.connection.rollback()
                    self.logger.warning("Transação revertida devido a erro.")
                except Error as rb_e:
                    self.logger.error(f"Erro ao reverter transação: {rb_e}")

            return None
        except Exception as e:
            self.logger.error(f"Um erro inesperado ocorreu: {e}")
            return None

    def table_exists(self, table_name, database_name=None):
        try:
            """
            Verifica se uma tabela existe na base de dados atual ou especificada.
            Args:
                table_name (str): O nome da tabela a verificar.
                database_name (str, optional): O nome da base de dados. Se None, usa a base de dados padrão da conexão.
            Returns:
                bool: True se a tabela existir, False caso contrário.
            """
            if database_name is None:
                database_name = self.DB_CONFIG.get('database')
                if not database_name:
                    self.logger.error("Nome da base de dados não especificado e não encontrado na configuração DB_CONFIG.")
                    return False

            query = """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s
                    AND table_name = %s
                ) AS table_exists;
            """
            params = (database_name, table_name)

            result = self.execute_query(query, params)

            if result is not None and len(result) > 0:
                return result[0]['table_exists'] == 1
            else:
                self.logger.error(f"Não foi possível verificar a existência da tabela '{table_name}'. Resultado da query: {result}")
                return False
        except Exception as e:
            self.logger.error(f"Um erro inesperado ocorreu: {e}")
            return False

    def get_dataframe(self, table_name):
        """
        Busca todos os dados de uma tabela e retorna um DataFrame Pandas correspondente.
        Args:
            table_name (str): O nome da tabela da qual buscar os dados.
        Returns:
            pd.DataFrame or None: O DataFrame Pandas com os dados da tabela, ou None em caso de erro.
        """
        self.logger.info(f"A obter dados da tabela '{table_name}' para um DataFrame...")
        query = f"SELECT * FROM `{self.DB_CONFIG['database']}`.`{table_name}`;"

        try:
            data = self.execute_query(query)

            if data is not None:
                if data:
                    df = pd.DataFrame(data)
                    self.logger.info(f"DataFrame para '{table_name}' criado com {len(df)} linhas.")
                    return df
                else:
                    self.logger.warning(f"Nenhum dado encontrado na tabela '{table_name}'. Retornando um DataFrame vazio.")
                    return pd.DataFrame()
            else:
                self.logger.error(f"Erro ao buscar dados da tabela '{table_name}' através de execute_query. Não foi possível criar o DataFrame.")
                return None
        except Exception as e:
            self.logger.error(f"Ocorreu um erro inesperado ao criar o DataFrame para a tabela '{table_name}': {e}")
            return None

    def get_dataframe_from_query(self, query=None, params=None):
        """
        Executa uma query SQL e retorna os resultados como um DataFrame Pandas.
        Args:
            query (str): A query SQL a ser executada.
            params (tuple, optional): Parâmetros para a query, se houver. Defaults to None.
        Returns:
            pd.DataFrame or None: O DataFrame Pandas com os resultados da query, ou None em caso de erro.
        """
        self.logger.info(f"A obter dados de query SQL para um DataFrame...")

        if query is None:
          query = self.joint_query

        try:

            data = self.execute_query(query, params=params)

            if data is not None:
                if data:  # Se a lista não estiver vazia
                    df = pd.DataFrame(data)
                    self.logger.info(f"DataFrame para a query criado com {len(df)} linhas.")
                    return df
                else:
                    self.logger.warning("Nenhum dado encontrado para a query. Retornando um DataFrame vazio.")
                    return pd.DataFrame()  # Retorna um DataFrame vazio se não houver dados
            else:
                self.logger.error("Erro ao executar query. Não foi possível criar o DataFrame.")
                return None
        except Exception as e:
            self.logger.error(f"Ocorreu um erro inesperado ao criar o DataFrame a partir da query: {e}")
            return None

    def setup_database(self,  table_configs: dict = None): # type: ignore
        """
        Configura as tabelas na base de dados com base num dicionário de configurações.
        Verifica a existência de cada tabela e a cria se não existir.
        Args:
            table_configurations (dict): Um dicionário onde as chaves são os nomes das tabelas
                                         e os valores são as queries SQL para criá-las.
        Returns:
            bool: True se todas as tabelas forem configuradas com sucesso, False caso contrário.
        """
        if table_configs is None:
          table_configs = self.TABLE_CONFIGURATIONS

        self.logger.info(f"Table Configs: '{table_configs}'")
        try:
            if not self.connect():
                self.logger.error("Não foi possível conectar à base de dados para configurar.")
                return False

            
            database_name = self.DB_CONFIG['database']

            for table_name, create_query_template in table_configs.items():
                create_table_query = create_query_template.format(database_name=database_name)

                if not self.table_exists(table_name):
                    self.logger.info(f"Tabela '{table_name}' não encontrada. A criar a tabela...")
                    if self.execute_query(create_table_query, commit=True):
                        self.logger.info(f"Tabela '{table_name}' criada com sucesso.")
                    else:
                        self.logger.error(f"Erro ao criar a tabela '{table_name}'.")
                        return False
                else:
                    self.logger.info(f"Tabela '{table_name}' já existe. Verificando dados...")

                check_table_data_query = f"SELECT COUNT(*) as count FROM `{database_name}`.`{table_name}`;"
                result_count = self.execute_query(check_table_data_query)

                if result_count is None:
                    self.logger.error(f"Erro ao verificar contagem de registos na tabela '{table_name}'.")
                    return False

                if result_count[0]['count'] == 0:
                    self.logger.warning(f"Tabela '{table_name}' vazia.")

            self.logger.info("Todas as tabelas configuradas na base de dados com sucesso.")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao configurar base de dados: {e}")
            return False

class Distance:
    """
    Collection of functions for geographic distance calculations and regional classification
    specific to Portugal.

    # Sumário estatístico
    summary = Distance.get_coordinates_summary(df, 'latitude', 'longitude')

    """
    

    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth using the Haversine formula.
        
        Args:
            lat1: Latitude of first point in decimal degrees
            lon1: Longitude of first point in decimal degrees
            lat2: Latitude of second point in decimal degrees
            lon2: Longitude of second point in decimal degrees
            
        Returns:
            float: Distance in kilometers
            
        Example:
            >>> Distance.haversine_distance(38.7223, -9.1393, 41.1579, -8.6291)
            274.23
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # Radius of Earth in kilometers
        R = 6371
        
        return R * c
    
    @classmethod
    def haversine_distance_from_row(
        cls, 
        row: pd.Series,
        lat1_col: str = 'latitude_casa',
        lon1_col: str = 'longitude_casa',
        lat2_col: str = 'latitude_jogo',
        lon2_col: str = 'longitude_jogo'
    ) -> Optional[float]:
        """
        Calculate the great circle distance between two points using data from a pandas DataFrame row.
        
        Args:
            row: pandas Series (DataFrame row) containing coordinate data
            lat1_col: Name of the column containing origin latitude (default: 'latitude_casa')
            lon1_col: Name of the column containing origin longitude (default: 'longitude_casa')
            lat2_col: Name of the column containing destination latitude (default: 'latitude_jogo')
            lon2_col: Name of the column containing destination longitude (default: 'longitude_jogo')
            
        Returns:
            Optional[float]: Distance in kilometers, or None if any coordinate is invalid/missing
            
        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'lat_origem': [38.7223, 40.4168],
            ...     'lon_origem': [-9.1393, -3.7038],
            ...     'lat_destino': [41.1579, 48.8566],
            ...     'lon_destino': [-8.6291, 2.3522]
            ... })
            >>> df['distance'] = df.apply(
            ...     lambda r: MyClass.haversine_distance_from_row(
            ...         r, 'lat_origem', 'lon_origem', 'lat_destino', 'lon_destino'
            ...     ), 
            ...     axis=1
            ... )
        """
        try:
            # Validate that all required columns exist in the row
            missing_cols = []
            for col in [lat1_col, lon1_col, lat2_col, lon2_col]:
                if col not in row.index:
                    missing_cols.append(col)
            
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} in row. Returning None.")
                return None
            
            # Extract coordinates and validate they are not None/NaN
            lat1 = row[lat1_col]
            lon1 = row[lon1_col]
            lat2 = row[lat2_col]
            lon2 = row[lon2_col]
            
            # Check for None or NaN values
            if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
                return None
            
            # Convert to float and validate numeric values
            lat1 = float(lat1)
            lon1 = float(lon1)
            lat2 = float(lat2)
            lon2 = float(lon2)
            
            # Validate coordinate ranges
            if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
                print(f"Warning: Invalid latitude values (must be between -90 and 90). Returning None.")
                return None
            
            if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
                print(f"Warning: Invalid longitude values (must be between -180 and 180). Returning None.")
                return None
            
            # Calculate and return distance
            return cls.haversine_distance(lat1, lon1, lat2, lon2)
            
        except (ValueError, TypeError) as e:
            print(f"Warning: Error converting coordinates to numeric values: {e}. Returning None.")
            return None
        except Exception as e:
            print(f"Warning: Unexpected error calculating distance: {e}. Returning None.")
            return None

class LigaSchedule:
    """
    Classe para gerir e validar slots das ligas portuguesas.

    Fornece utilitários para:
    - obter slots por liga / por dia
    - validar um slot e corrigir para o slot mais próximo
    - processar DataFrames de forma robusta
    """

    # Mapeamento dos códigos dos dias da semana (0=Segunda...6=Domingo)
    # Reflete o comportamento do weekday do Python/Pandas (0=Monday...6=Sunday)
    WEEKDAY_MAP_PT_TO_EN: Dict[str, int] = {
        "Segunda": 0, "Terça": 1, "Quarta": 2, "Quinta": 3,
        "Sexta": 4, "Sábado": 5, "Domingo": 6
    }
    
    WEEKDAY_MAP_EN_TO_INT: Dict[str, int] = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    
    # Adicionando um mapeamento auxiliar para facilitar a tradução de PT para INT
    WEEKDAY_NAME_MAP: Dict[str, int] = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
        "Segunda": 0, "Terça": 1, "Quarta": 2, "Quinta": 3,
        "Sexta": 4, "Sábado": 5, "Domingo": 6
    }

    LIGA_SLOTS: Dict[str, Dict[int, List[dt_time]]] = {
        "Liga Portugal Betclic": {
            4: [dt_time(20, 15)],
            5: [dt_time(15, 30), dt_time(18, 0), dt_time(20, 30)],
            6: [dt_time(15, 30), dt_time(18, 0), dt_time(20, 30)],
            0: [dt_time(20, 15)]
        },
        "Liga Portugal Meu Super": {
            4: [dt_time(18, 0)],
            5: [dt_time(11, 0), dt_time(14, 0), dt_time(15, 30)],
            6: [dt_time(11, 0), dt_time(14, 0), dt_time(15, 30)],
            0: [dt_time(18, 0)]
        }
    }

    @classmethod
    def get_slots_by_liga(cls, liga_name: str) -> Dict[int, List[dt_time]]:
        """Retorna slots para uma liga específica."""
        return cls.LIGA_SLOTS.get(liga_name, {})

    @classmethod
    def get_slots_by_day(cls, weekday: int) -> Dict[str, List[dt_time]]:
        result: Dict[str, List[dt_time]] = {}
        for liga_name, liga_slots in cls.LIGA_SLOTS.items():
            if weekday in liga_slots:
                result[liga_name] = liga_slots[weekday]
        return result

    @classmethod
    def get_specific_slot(cls, liga_name: str, weekday: int, slot_index: int = 0) -> Optional[dt_time]:
        liga_slots = cls.LIGA_SLOTS.get(liga_name, {})
        day_slots = liga_slots.get(weekday, [])
        if 0 <= slot_index < len(day_slots):
            return day_slots[slot_index]
        return None

    @classmethod
    def list_all_slots(cls) -> None:
        # Atualização para usar os nomes em inglês
        weekdays = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                    4: "Friday", 5: "Saturday", 6: "Sunday"}
        for liga_name, liga_slots in cls.LIGA_SLOTS.items():
            print(f"\n{liga_name}:")
            for weekday, slots in sorted(liga_slots.items()):
                day_name = weekdays.get(weekday, str(weekday))
                slots_str = ", ".join([s.strftime("%H:%M") for s in slots])
                print(f"  {day_name} ({weekday}): {slots_str}")

    @classmethod
    def validate_or_correct_slot(cls, liga: str, weekday: int, hora: Union[str, dt_time, datetime]) -> Tuple[int, dt_time]:
        """
        Valida um slot; se inválido retorna o slot mais próximo (dentro da mesma liga).
        Aceita hora como string (HH:MM ou HH:MM:SS), datetime.time, datetime.datetime ou pandas timestamp.
        """
        # Normalizar hora para objeto time
        hora_obj: Optional[dt_time] = None

        if isinstance(hora, str):
            raw = hora.strip()
            # Tentar parse com formatos conhecidos
            for fmt in ("%H:%M", "%H:%M:%S"):
                try:
                    hora_obj = datetime.strptime(raw, fmt).time()
                    break
                except Exception:
                    hora_obj = None
            if hora_obj is None:
                # Tentar pandas parser (mais flexível)
                parsed = pd.to_datetime(raw, errors='coerce')
                if not pd.isna(parsed):
                    hora_obj = parsed.time()
                else:
                    raise ValueError(f"Formato de hora inválido: {hora}. Use 'HH:MM' ou 'HH:MM:SS' ou um timestamp válido")
        elif isinstance(hora, datetime):
            hora_obj = hora.time()
        elif hasattr(hora, 'hour') and hasattr(hora, 'minute'):
            # inclui datetime.time e pandas Timestamp
            hora_obj = hora
        else:
            raise ValueError(f"Tipo de hora não suportado: {type(hora)}")

        if liga not in cls.LIGA_SLOTS:
            return weekday, hora_obj  # Retorna valores originais

        liga_slots = cls.LIGA_SLOTS[liga]

        if weekday in liga_slots and hora_obj in liga_slots[weekday]:
            return weekday, hora_obj

        # Se não existe, encontrar o slot mais próximo
        return cls._find_nearest_slot(liga_slots, weekday, hora_obj)

    @staticmethod
    def _find_nearest_slot(liga_slots: Dict[int, List[dt_time]], target_weekday: int, target_time: dt_time) -> Tuple[int, dt_time]:
        min_diff = float('inf')
        best_slot: Optional[Tuple[int, dt_time]] = None
        target_minutes = target_time.hour * 60 + target_time.minute

        for weekday, slots in liga_slots.items():
            for slot_time in slots:
                slot_minutes = slot_time.hour * 60 + slot_time.minute
                day_diff = abs(weekday - target_weekday)
                day_diff = min(day_diff, 7 - day_diff)
                total_diff = day_diff * 24 * 60 + abs(slot_minutes - target_minutes)
                if total_diff < min_diff:
                    min_diff = total_diff
                    best_slot = (weekday, slot_time)

        if best_slot:
            return best_slot
        return (target_weekday, target_time)

    @classmethod
    def process_dataframe_slots(cls, df: pd.DataFrame, liga_col: str, weekday_col: str, hora_col: str) -> pd.DataFrame:
        """
        Processa um DataFrame validando e corrigindo slots das ligas.
        Corrige automaticamente os nomes dos dias da semana (em inglês ou português)
        para o formato de número inteiro (0=Monday/Segunda...6=Sunday/Domingo).

        Retorna um DataFrame com colunas adicionais:
            - 'valid_slot': bool
            - 'slot_weekday': int
            - 'slot_hour': datetime.time
        """
        df_result = df.copy()

        def _process_row(row):
            original_weekday_input = row[weekday_col]
            original_weekday: Union[int, None] = None
            
            # --- LÓGICA DE TRADUÇÃO/CORREÇÃO DO DIA DA SEMANA ---
            if isinstance(original_weekday_input, (int, float)):
                # Se for número (float/int), tenta converter para int
                try:
                    original_weekday = int(original_weekday_input) % 7 # Assegura que está no intervalo [0-6]
                except (ValueError, TypeError):
                    pass # Deixa como None se a conversão falhar
            elif isinstance(original_weekday_input, str):
                # Se for nome (string), tenta traduzir
                day_name = original_weekday_input.strip().title()
                original_weekday = cls.WEEKDAY_NAME_MAP.get(day_name, None)
            
            # Se não conseguiu obter um weekday válido, não é possível prosseguir
            if original_weekday is None:
                return pd.Series([False, row.get(weekday_col, None), row.get(hora_col, None)], 
                                 index=['valid_slot', 'slot_weekday', 'slot_hour'])
            
            # --- CONTINUAÇÃO DA LÓGICA ORIGINAL ---
            original_time_value = row[hora_col]
            liga = row[liga_col]

            try:
                corrected_weekday, corrected_time = cls.validate_or_correct_slot(liga, original_weekday, original_time_value)
                
                # Normalizar original_time para comparação
                original_time_obj: Optional[dt_time] = None
                if isinstance(original_time_value, str):
                    parsed = pd.to_datetime(original_time_value, errors='coerce')
                    if not pd.isna(parsed):
                        original_time_obj = parsed.time()
                elif isinstance(original_time_value, datetime):
                    original_time_obj = original_time_value.time()
                elif hasattr(original_time_value, 'hour') and hasattr(original_time_value, 'minute'):
                    original_time_obj = original_time_value
                
                valid_slot = (original_weekday == corrected_weekday and original_time_obj == corrected_time)
                return pd.Series([valid_slot, corrected_weekday, corrected_time], index=['valid_slot', 'slot_weekday', 'slot_hour'])
            
            except Exception as e:
                # Em caso de erro, devolve valores originais e marca como inválido
                return pd.Series([False, original_weekday, row.get(hora_col, None)], 
                                 index=['valid_slot', 'slot_weekday', 'slot_hour'])

        processed = df_result.apply(_process_row, axis=1)
        df_result[['valid_slot', 'slot_weekday', 'slot_hour']] = processed
        return df_result

    @classmethod
    def calcular_data_hora_final(cls, df):
        """
        Calcula o campo final de data e hora a partir da data inicial e do slot final.

        Argumentos:
        df (pd.DataFrame): O DataFrame contendo as colunas 'Data_hora' e 'final_slot'.

        Retorna:
        pd.DataFrame: O DataFrame com a nova coluna 'Data_hora_final'.
        """
        # Certifique-se de que a coluna Data_hora é do tipo datetime
        df['date_time'] = pd.to_datetime(df['date_time'])

        # Extrai o dia da semana (como número) e a hora do campo final_slot
        df['final_weekday_num'] = df['final_slot'].str.split(' ').str[0].astype(int)
        df['final_hour'] = df['final_slot'].str.split(' ').str[1]

        # Mapeia os dias da semana para um número para calcular a diferença
        df['initial_weekday_num'] = df['date_time'].dt.weekday

        # Calcula a diferença de dias. Adiciona 7 se a diferença for negativa (para avançar uma semana)
        df['delta_days'] = df['final_weekday_num'] - df['initial_weekday_num']
        df['delta_days'] = df.apply(
            lambda row: row['delta_days'] + 7 if row['delta_days'] < 0 else row['delta_days'],
            axis=1
        )

        # Cria a nova coluna de data e hora
        df['final_date_time'] = df['date_time'] + pd.to_timedelta(df['delta_days'], unit='D')
        df['final_date_time'] = df.apply(
            lambda row: row['final_date_time'].replace(
                hour=int(row['final_hour'].split(':')[0]),
                minute=int(row['final_hour'].split(':')[1])
            ),
            axis=1
        )
        df = df.drop(columns=['final_hour', 'initial_weekday_num', 'delta_days'])

        return df

class WeatherInformation:
    """
    Classe para obter e processar dados meteorológicos.
    """
    
    def obter_dados_meteorologicos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtém dados meteorológicos para cada linha do DataFrame.
        Ignora linhas com Data_Hora inválida ou coordenadas ausentes.
        """
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session) # type: ignore

        df['temperatura_aparente_media'] = None
        df['precipitacao_media'] = None
        df['velocidade_vento_media'] = None

        for index, row in df.iterrows():
            try:
                # Validar Data_Hora
                date_time_obj = pd.to_datetime(row['Data_Hora'], errors='coerce')
                if pd.isna(date_time_obj):
                    print(f"Warning: linha {index} tem Data_Hora inválida (NaT). Saltando.")
                    continue

                data = date_time_obj.strftime('%Y-%m-%d')
                hour_of_interest = date_time_obj.hour

                # Validar coordenadas
                if pd.isna(row['latitude_jogo']) or pd.isna(row['longitude_jogo']):
                    print(f"Warning: linha {index} sem coordenadas válidas. Saltando.")
                    continue

                params = {
                    "latitude": float(row['latitude_jogo']),
                    "longitude": float(row['longitude_jogo']),
                    "start_date": data,
                    "end_date": data,
                    "hourly": ["apparent_temperature", "precipitation", "wind_speed_10m"],
                    "timezone": "Europe/London",
                }

                responses = openmeteo.weather_api(
                    url="https://archive-api.open-meteo.com/v1/archive",
                    params=params
                )
                response = responses[0]
                hourly = response.Hourly()

                # Extrair dados horários
                hourly_apparent_temperature = hourly.Variables(0).ValuesAsNumpy() # type: ignore
                hourly_precipitation = hourly.Variables(1).ValuesAsNumpy() # type: ignore
                hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy() # type: ignore

                # Janela de tempo
                start_hour_index = max(0, hour_of_interest - 1)
                end_hour_index = min(24, hour_of_interest + 3)

                # Calcular médias
                df.loc[index, 'temperatura_aparente_media'] = hourly_apparent_temperature[start_hour_index:end_hour_index].mean() # type: ignore
                df.loc[index, 'precipitacao_media'] = hourly_precipitation[start_hour_index:end_hour_index].mean() # type: ignore
                df.loc[index, 'velocidade_vento_media'] = hourly_wind_speed_10m[start_hour_index:end_hour_index].mean() # type: ignore

                time.sleep(2)

            except Exception as e:
                print(f"Erro ao obter dados meteorológicos para a linha {index}: {e}")

        return df

class DataVisualization:
    """
    Class for creating data visualizations from DataFrames.
    Supports single and two-variable charts with complete validations.
    """

    def __init__(self,
                 figsize: tuple[int, int] = (10, 6),
                 style: Literal['white', 'dark', 'whitegrid', 'darkgrid', 'ticks'] = 'whitegrid'):
        """
        Initialize the visualization class.

        Args:
            figsize: Tuple with figure size (width, height)
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        """
        self.figsize = figsize
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _validate_dataframe(self, data: Union[pd.DataFrame, pd.Series, Dict]) -> pd.DataFrame:
        """Validate and convert data to DataFrame."""
        if isinstance(data, pd.Series):
            return data.to_frame()
        elif isinstance(data, dict):
            try:
                return pd.DataFrame(data)
            except Exception as e:
                raise ValueError(f"Error converting dictionary to DataFrame: {e}")
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            raise TypeError("Data must be DataFrame, Series or dictionary")

    def _validate_column(self, df: pd.DataFrame, column: str, col_type: str = "column") -> None:
        """Validate if column exists in DataFrame."""
        if column not in df.columns:
            raise ValueError(f"{col_type} '{column}' not found. Available columns: {list(df.columns)}")

    def _clean_data(self, df: pd.DataFrame, columns: List[str], dropna: bool = True) -> pd.DataFrame:
        """Remove null and invalid values from specified columns."""
        df_clean = df[columns].copy()

        if dropna:
            initial_size = len(df_clean)
            df_clean = df_clean.dropna()
            removed = initial_size - len(df_clean)
            if removed > 0:
                warnings.warn(f"{removed} rows with null values were removed")

        return df_clean

    # ------------------------------------------------------------------
    # Visualization methods
    # ------------------------------------------------------------------

    def bar_chart(self,
                  data: Union[pd.DataFrame, pd.Series, Dict],
                  column: str,
                  title: Optional[str] = None,
                  xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None,
                  color: str = 'steelblue',
                  horizontal: bool = False,
                  top_n: Optional[int] = None,
                  sort_values: bool = True) -> Figure:
        """Create a bar chart for a single categorical variable."""
        df = self._validate_dataframe(data)
        self._validate_column(df, column)

        df_clean = self._clean_data(df, [column])
        value_counts = df_clean[column].value_counts()

        if sort_values:
            value_counts = value_counts.sort_values(ascending=not horizontal)
        if top_n is not None:
            value_counts = value_counts.head(top_n)

        fig, ax = plt.subplots(figsize=self.figsize)

        if horizontal:
            ax.barh(value_counts.index.astype(str), value_counts.to_numpy(), color=color)
            ax.set_xlabel(ylabel or 'Count')
            ax.set_ylabel(xlabel or column)
        else:
            ax.bar(value_counts.index.astype(str), value_counts.to_numpy(), color=color)
            ax.set_xlabel(xlabel or column)
            ax.set_ylabel(ylabel or 'Count')
            plt.xticks(rotation=45, ha='right')

        ax.set_title(title or f'Distribution of {column}')
        plt.tight_layout()
        return fig

    def pie_chart(self,
                  data: Union[pd.DataFrame, pd.Series, Dict],
                  column: str,
                  title: Optional[str] = None,
                  autopct: str = '%1.1f%%',
                  top_n: Optional[int] = None,
                  colors: Optional[List[str]] = None) -> Figure:
        """Create a pie chart for a single categorical variable."""
        df = self._validate_dataframe(data)
        self._validate_column(df, column)

        df_clean = self._clean_data(df, [column])
        value_counts = df_clean[column].value_counts()

        if top_n is not None and len(value_counts) > top_n:
            top_values = value_counts.head(top_n)
            other_sum = value_counts.iloc[top_n:].sum()
            value_counts = pd.concat([top_values, pd.Series({'Other': other_sum})])

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.pie(
            value_counts.to_numpy(),
            labels=list(value_counts.index.astype(str)),
            autopct=autopct,
            colors=colors,
            startangle=90
)
        ax.set_title(title or f'Distribution of {column}')
        plt.tight_layout()
        return fig

    def histogram(self,
                  data: Union[pd.DataFrame, pd.Series, Dict],
                  column: str,
                  bins: Union[int, str] = 'auto',
                  title: Optional[str] = None,
                  xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None,
                  color: str = 'steelblue',
                  kde: bool = False) -> Figure:
        """Create a histogram for a numerical variable."""
        df = self._validate_dataframe(data)
        self._validate_column(df, column)

        df_clean = self._clean_data(df, [column])
        if not pd.api.types.is_numeric_dtype(df_clean[column]):
            raise TypeError(f"Column '{column}' must be numeric for histogram")

        fig, ax = plt.subplots(figsize=self.figsize)
        if kde:
            sns.histplot(data=df_clean, x=column, bins=bins, color=color, kde=True, ax=ax)
        else:
            ax.hist(df_clean[column].to_numpy(), bins=bins, color=color, edgecolor='black', alpha=0.7)

        ax.set_xlabel(xlabel or column)
        ax.set_ylabel(ylabel or 'Frequency')
        ax.set_title(title or f'Distribution of {column}')
        plt.tight_layout()
        return fig

    def histplot(self,
                 data: Union[pd.DataFrame, Dict],
                 column: str,
                 hue: Optional[str] = None,
                 bins: Union[int, str] = 'auto',
                 stat: Literal['count', 'frequency', 'probability', 'percent', 'density'] = 'count',
                 multiple: Literal['layer', 'dodge', 'stack', 'fill'] = 'layer',
                 element: Literal['bars', 'step', 'poly'] = 'bars',
                 kde: bool = False,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 palette: Optional[str] = None) -> Figure:
        """Create a histogram using seaborn's histplot with advanced options."""
        df = self._validate_dataframe(data)
        self._validate_column(df, column)

        columns_to_clean = [column]
        if hue:
            self._validate_column(df, hue)
            columns_to_clean.append(hue)

        df_clean = self._clean_data(df, columns_to_clean)
        if not pd.api.types.is_numeric_dtype(df_clean[column]):
            raise TypeError(f"Column '{column}' must be numeric for histplot")

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.histplot(data=df_clean, x=column, hue=hue, bins=bins, stat=stat,
                     multiple=multiple, element=element, kde=kde, palette=palette, ax=ax)

        ax.set_xlabel(xlabel or column)
        ax.set_ylabel(ylabel or stat.capitalize())
        ax.set_title(title or f'Distribution of {column}')
        if hue:
            ax.legend(title=hue)
        plt.tight_layout()
        return fig

    def scatter_plot(self,
                     data: Union[pd.DataFrame, Dict],
                     x_column: str,
                     y_column: str,
                     title: Optional[str] = None,
                     xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     color: str = 'steelblue',
                     size: Union[int, str] = 50,
                     alpha: float = 0.6) -> Figure:
        """Create a scatter plot for two numerical variables."""
        df = self._validate_dataframe(data)
        self._validate_column(df, x_column, "X column")
        self._validate_column(df, y_column, "Y column")

        df_clean = self._clean_data(df, [x_column, y_column])
        if not pd.api.types.is_numeric_dtype(df_clean[x_column]) or not pd.api.types.is_numeric_dtype(df_clean[y_column]):
            raise TypeError("Both x and y columns must be numeric for scatter plot")

        fig, ax = plt.subplots(figsize=self.figsize)
        if isinstance(size, str):
            self._validate_column(df, size, "Size column")
            sizes = df_clean[size].to_numpy()
            ax.scatter(df_clean[x_column], df_clean[y_column], c=color, s=sizes, alpha=alpha)
        else:
            ax.scatter(df_clean[x_column], df_clean[y_column], c=color, s=size, alpha=alpha)

        ax.set_xlabel(xlabel or x_column)
        ax.set_ylabel(ylabel or y_column)
        ax.set_title(title or f'{y_column} vs {x_column}')
        plt.tight_layout()
        return fig

    def line_chart(self,
                   data: Union[pd.DataFrame, Dict],
                   x_column: str,
                   y_column: Union[str, List[str]],
                   title: Optional[str] = None,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   marker: str = 'o',
                   linestyle: str = '-') -> Figure:
        """Create a line chart for one or multiple numerical variables."""
        df = self._validate_dataframe(data)
        self._validate_column(df, x_column, "X column")

        y_columns = [y_column] if isinstance(y_column, str) else y_column
        for col in y_columns:
            self._validate_column(df, col, "Y column")

        df_clean = self._clean_data(df, [x_column] + y_columns)
        fig, ax = plt.subplots(figsize=self.figsize)

        for col in y_columns:
            ax.plot(df_clean[x_column], df_clean[col], marker=marker, linestyle=linestyle, label=col)

        ax.set_xlabel(xlabel or x_column)
        ax.set_ylabel(ylabel or 'Value')
        ax.set_title(title or 'Line Chart')
        if len(y_columns) > 1:
            ax.legend()
        plt.tight_layout()
        return fig

    def grouped_bar_chart(self,
                          data: Union[pd.DataFrame, Dict],
                          category_column: str,
                          value_column: str,
                          group_column: str,
                          title: Optional[str] = None,
                          xlabel: Optional[str] = None,
                          ylabel: Optional[str] = None) -> Figure:
        """Create a grouped bar chart for two categorical variables."""
        df = self._validate_dataframe(data)
        self._validate_column(df, category_column)
        self._validate_column(df, value_column)
        self._validate_column(df, group_column)

        df_clean = self._clean_data(df, [category_column, value_column, group_column])
        pivot_df = df_clean.pivot_table(values=value_column, index=category_column,
                                        columns=group_column, aggfunc='sum', fill_value=0)

        fig, ax = plt.subplots(figsize=self.figsize)
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_xlabel(xlabel or category_column)
        ax.set_ylabel(ylabel or value_column)
        ax.set_title(title or f'{value_column} by {category_column} and {group_column}')
        ax.legend(title=group_column)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def box_plot(self,
                 data: Union[pd.DataFrame, Dict],
                 column: Optional[str] = None,
                 group_column: Optional[str] = None,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None) -> Figure:
        """Create a box plot for numerical variable(s)."""
        df = self._validate_dataframe(data)
        fig, ax = plt.subplots(figsize=self.figsize)

        if group_column:
            self._validate_column(df, group_column)

            if column is None:
                raise ValueError("Parameter 'column' must be provided when 'group_column' is specified.")

            self._validate_column(df, column)

            # Apenas colunas não nulas
            df_clean = self._clean_data(df, [c for c in [column, group_column] if c is not None])

            df_clean.boxplot(column=column, by=group_column, ax=ax)
            ax.set_xlabel(str(xlabel or group_column))
            ax.set_ylabel(str(ylabel or column))
            plt.suptitle('')
            ax.set_title(title or f'{column} by {group_column}')

        else:
            if column:
                self._validate_column(df, column)
                df_clean = self._clean_data(df, [column])
                ax.boxplot(df_clean[column].to_numpy())
                ax.set_ylabel(str(ylabel or column))
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError("No numeric columns found in DataFrame")

                df_clean = self._clean_data(df, numeric_cols)
                ax.boxplot(df_clean[numeric_cols].to_numpy())
                ax.set_xticks(range(1, len(numeric_cols) + 1))
                ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
                ax.set_ylabel(str(ylabel or 'Value'))

            ax.set_title(title or 'Box Plot')


        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig




# Utility functions (not in classes)

def infer_timezone(row):
    """Infer timezone for stadiums based on coordinates (Açores vs rest of Portugal)."""
    lat, lon = row['latitud'], row['longitud']
    regiao = str(row.get('simple_region', '')).lower() + ' ' + str(row.get('detailed_region', '')).lower()

    # Garantir que está dentro das coordenadas de Portugal
    if pd.notna(lat) and pd.notna(lon) and (32 <= lat <= 43) and (-32 <= lon <= -6):
        # Açores → Atlantic/Azores
        if 'açor' in regiao or (-32 < lon < -25):
            return 'Atlantic/Azores'
        # Restante território → Europe/Lisbon
        else:
            return 'Europe/Lisbon'
    else:
        # Fora das coordenadas de Portugal → indefinido
        return None

def print_separator(title: str = "", char: str = "=", length: int = 50) -> None:
    """
    Print a separator line with optional title.

    Args:
        title: Title to display in the middle
        char: Character to use for separator
        length: Length of the separator line
    """
    if title:
        title = f" {title} "
        print(title.center(length, char))
    else:
        print(char * length)

def memory_usage(df: pd.DataFrame, detailed: bool = False) -> None:
    """
    Print memory usage of DataFrame.

    Args:
        df: DataFrame to analyze
        detailed: Whether to show detailed breakdown
    """
    total_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Total memory usage: {total_mb:.2f} MB")

    if detailed:
        print("\nMemory usage by column:")
        mem_usage = df.memory_usage(deep=True)
        for col, usage in mem_usage.items():
            print(f"{col}: {usage / 1024**2:.2f} MB")

def obter_dados_meteorologicos(df):
    """
    Obtém dados meteorológicos para cada linha do DataFrame, focando em uma janela
    de tempo de 1 hora antes e 2 horas depois do horário de Data_Hora.
    Calcula apenas os valores médios para essa janela.
    """
    # Configura o cliente da API Open-Meteo com cache e retentativa em caso de erro
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session) # type: ignore

    # Adiciona colunas para os dados meteorológicos calculados (apenas a média)
    df['temperatura_aparente_media'] = None
    df['precipitacao_media'] = None
    df['velocidade_vento_media'] = None

    # Itera sobre cada linha para obter dados meteorológicos
    for index, row in df.iterrows():
        try:
            # Converte a coluna Data_Hora para o formato datetime, se ainda não estiver
            date_time_obj = pd.to_datetime(row['Data_Hora'])
            data = date_time_obj.strftime('%Y-%m-%d')
            hour_of_interest = date_time_obj.hour

            params = {
                "latitude": row['Latitude_Casa'],
                "longitude": row['Longitude_Casa'],
                "start_date": data,
                "end_date": data,
                "hourly": ["apparent_temperature", "precipitation", "wind_speed_10m"],
                "timezone": "Europe/London",
            }

            responses = openmeteo.weather_api(url="https://archive-api.open-meteo.com/v1/archive", params=params)
            response = responses[0]
            hourly = response.Hourly()

            # Obtém os dados horários
            hourly_apparent_temperature = hourly.Variables(0).ValuesAsNumpy() # type: ignore
            hourly_precipitation = hourly.Variables(1).ValuesAsNumpy() # type: ignore
            hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy() # type: ignore

            # Define a janela de tempo de 4 horas (2h antes e 2h depois)
            # O índice final do slice em Python é exclusivo, então adicionamos 1
            start_hour_index = max(0, hour_of_interest - 1)
            end_hour_index = min(24, hour_of_interest + 3)

            # Extrai os dados para a janela de tempo
            windowed_apparent_temperature = hourly_apparent_temperature[start_hour_index:end_hour_index].astype(float)
            windowed_precipitation = hourly_precipitation[start_hour_index:end_hour_index].astype(float)
            windowed_wind_speed_10m = hourly_wind_speed_10m[start_hour_index:end_hour_index].astype(float)

            # Calcula os valores médios das variáveis na janela de tempo
            df.at[index, 'temperatura_aparente_media'] = windowed_apparent_temperature.mean()
            df.at[index, 'precipitacao_media'] = windowed_precipitation.mean()
            df.at[index, 'velocidade_vento_media'] = windowed_wind_speed_10m.mean()
            time.sleep(2)  # Espera 2 segundos entre as chamadas para evitar sobrecarga na API

        except Exception as e:
            print(f"Erro ao obter dados meteorológicos para a linha {index}: {e}")

    return df