"""
Parquet Handler for Timeseries Data
각 CSV 파일의 timeseries 분석 결과를 Parquet 형식으로 저장/로드
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class TimeseriesParquetHandler:
    """Timeseries 데이터를 Parquet 파일로 저장/로드하는 클래스"""

    @staticmethod
    def save_timeseries(
        output_path: str,
        timeseries_df: pd.DataFrame,
        metadata: Dict[str, Any],
        statistics: Dict[str, Any]
    ):
        """
        Timeseries 데이터를 Parquet 파일로 저장

        Parameters
        ----------
        output_path : str
            저장할 Parquet 파일 경로
        timeseries_df : pd.DataFrame
            인덱스별 timeseries 데이터
        metadata : dict
            메타데이터 (csv_path, n_samples, duration_s 등)
        statistics : dict
            통계값 (mean, median, std, percentiles 등)
        """
        # DataFrame을 PyArrow Table로 변환
        table = pa.Table.from_pandas(timeseries_df)

        # Metadata와 Statistics를 JSON으로 직렬화
        custom_metadata = {
            'metadata': json.dumps(metadata),
            'statistics': json.dumps(statistics)
        }

        # 기존 schema metadata와 병합
        existing_metadata = table.schema.metadata or {}
        combined_metadata = {**existing_metadata, **custom_metadata}

        # Schema에 metadata 추가
        new_schema = table.schema.with_metadata(combined_metadata)
        table = table.cast(new_schema)

        # Parquet 파일로 저장
        pq.write_table(table, output_path, compression='snappy')

    @staticmethod
    def load_timeseries(
        parquet_path: str,
        load_timeseries: bool = True,
        load_metadata: bool = True,
        load_statistics: bool = True
    ) -> Dict[str, Any]:
        """
        Parquet 파일에서 Timeseries 데이터 로드

        Parameters
        ----------
        parquet_path : str
            Parquet 파일 경로
        load_timeseries : bool
            Timeseries DataFrame 로드 여부
        load_metadata : bool
            Metadata 로드 여부
        load_statistics : bool
            Statistics 로드 여부

        Returns
        -------
        dict
            {
                'timeseries': pd.DataFrame or None,
                'metadata': dict or None,
                'statistics': dict or None
            }
        """
        result = {
            'timeseries': None,
            'metadata': None,
            'statistics': None
        }

        # Parquet 파일 읽기
        table = pq.read_table(parquet_path)

        # Timeseries DataFrame 로드
        if load_timeseries:
            result['timeseries'] = table.to_pandas()

        # Metadata 로드
        if load_metadata or load_statistics:
            schema_metadata = table.schema.metadata or {}

            if load_metadata and b'metadata' in schema_metadata:
                result['metadata'] = json.loads(schema_metadata[b'metadata'].decode('utf-8'))

            if load_statistics and b'statistics' in schema_metadata:
                result['statistics'] = json.loads(schema_metadata[b'statistics'].decode('utf-8'))

        return result

    @staticmethod
    def get_metadata_only(parquet_path: str) -> Dict[str, Any]:
        """
        Parquet 파일에서 Metadata만 빠르게 로드 (timeseries 읽지 않음)

        Parameters
        ----------
        parquet_path : str
            Parquet 파일 경로

        Returns
        -------
        dict
            Metadata
        """
        parquet_file = pq.ParquetFile(parquet_path)
        schema_metadata = parquet_file.schema_arrow.metadata or {}

        if b'metadata' in schema_metadata:
            return json.loads(schema_metadata[b'metadata'].decode('utf-8'))
        else:
            return {}

    @staticmethod
    def get_statistics_only(parquet_path: str) -> Dict[str, Any]:
        """
        Parquet 파일에서 Statistics만 빠르게 로드 (timeseries 읽지 않음)

        Parameters
        ----------
        parquet_path : str
            Parquet 파일 경로

        Returns
        -------
        dict
            Statistics
        """
        parquet_file = pq.ParquetFile(parquet_path)
        schema_metadata = parquet_file.schema_arrow.metadata or {}

        if b'statistics' in schema_metadata:
            return json.loads(schema_metadata[b'statistics'].decode('utf-8'))
        else:
            return {}


class TimeseriesBuilder:
    """각 모듈의 분석 결과를 하나의 Timeseries DataFrame으로 결합"""

    def __init__(self, csv_path: str):
        """
        Parameters
        ----------
        csv_path : str
            원본 CSV 파일 경로
        """
        self.csv_path = os.path.abspath(csv_path)
        self.csv_filename = os.path.basename(csv_path)

        # 원본 CSV 읽기
        self.df = pd.read_csv(csv_path)
        self.n_samples = len(self.df)

        # Timeseries 데이터 저장용
        self.timeseries_data = {}

        # Statistics 저장용
        self.statistics = {}

        # Metadata
        self.metadata = {
            'csv_path': self.csv_path,
            'csv_filename': self.csv_filename,
            'n_samples': self.n_samples
        }

    def add_timeseries(self, key: str, data: pd.Series):
        """
        Timeseries 데이터 추가

        Parameters
        ----------
        key : str
            컬럼명
        data : pd.Series or array-like
            시계열 데이터
        """
        self.timeseries_data[key] = data

    def add_statistics(self, category: str, stats: Dict[str, Any]):
        """
        Statistics 추가

        Parameters
        ----------
        category : str
            카테고리명 (예: 'qos', 'uncertainty', 'safety')
        stats : dict
            통계값
        """
        if category not in self.statistics:
            self.statistics[category] = {}
        self.statistics[category].update(stats)

    def add_metadata(self, key: str, value: Any):
        """
        Metadata 추가

        Parameters
        ----------
        key : str
            키
        value : any
            값
        """
        self.metadata[key] = value

    def build(self) -> pd.DataFrame:
        """
        Timeseries DataFrame 생성

        Returns
        -------
        pd.DataFrame
            Timeseries 데이터
        """
        return pd.DataFrame(self.timeseries_data)

    def save(self, output_dir: str) -> str:
        """
        Parquet 파일로 저장

        Parameters
        ----------
        output_dir : str
            저장할 디렉토리

        Returns
        -------
        str
            저장된 파일 경로
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 파일명: 원본 CSV와 동일, 확장자만 .parquet
        output_filename = os.path.splitext(self.csv_filename)[0] + '.parquet'
        output_path = os.path.join(output_dir, output_filename)

        # DataFrame 생성
        timeseries_df = self.build()

        # Parquet 저장
        TimeseriesParquetHandler.save_timeseries(
            output_path=output_path,
            timeseries_df=timeseries_df,
            metadata=self.metadata,
            statistics=self.statistics
        )

        return output_path


def calculate_simple_statistics(series: pd.Series, prefix: str = "") -> Dict[str, float]:
    """
    Mean과 Std만 계산하는 간단한 통계 함수

    Parameters
    ----------
    series : pd.Series
        데이터
    prefix : str
        키 prefix (예: "delay_ms_")

    Returns
    -------
    dict
        {"mean": ..., "std": ...}
    """
    valid_data = series.dropna()

    if len(valid_data) == 0:
        return {
            f"{prefix}mean": None,
            f"{prefix}std": None
        }

    return {
        f"{prefix}mean": float(valid_data.mean()),
        f"{prefix}std": float(valid_data.std())
    }


def calculate_full_statistics(series: pd.Series, prefix: str = "") -> Dict[str, float]:
    """
    모든 통계값 계산 (Parquet 파일에 저장용)

    Parameters
    ----------
    series : pd.Series
        데이터
    prefix : str
        키 prefix

    Returns
    -------
    dict
        mean, median, std, min, max, p10, p25, p50, p75, p90, p95
    """
    valid_data = series.dropna()

    if len(valid_data) == 0:
        return {
            f"{prefix}mean": None,
            f"{prefix}median": None,
            f"{prefix}std": None,
            f"{prefix}min": None,
            f"{prefix}max": None,
            f"{prefix}p10": None,
            f"{prefix}p25": None,
            f"{prefix}p50": None,
            f"{prefix}p75": None,
            f"{prefix}p90": None,
            f"{prefix}p95": None
        }

    import numpy as np

    return {
        f"{prefix}mean": float(valid_data.mean()),
        f"{prefix}median": float(valid_data.median()),
        f"{prefix}std": float(valid_data.std()),
        f"{prefix}min": float(valid_data.min()),
        f"{prefix}max": float(valid_data.max()),
        f"{prefix}p10": float(np.percentile(valid_data, 10)),
        f"{prefix}p25": float(np.percentile(valid_data, 25)),
        f"{prefix}p50": float(np.percentile(valid_data, 50)),
        f"{prefix}p75": float(np.percentile(valid_data, 75)),
        f"{prefix}p90": float(np.percentile(valid_data, 90)),
        f"{prefix}p95": float(np.percentile(valid_data, 95))
    }
