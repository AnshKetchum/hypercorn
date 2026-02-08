from __future__ import annotations

import datetime
from typing import List, Optional, Iterator, Dict, Any

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict


class RunMeta(BaseModel):
    """Metadata about the execution of a submission."""
    model_config = ConfigDict(from_attributes=True)
    
    command: Optional[str] = None
    duration: Optional[float] = None
    exit_code: Optional[int] = None
    stderr: Optional[str] = None
    stdout: Optional[str] = None
    success: Optional[bool] = None


class RunSystemInfo(BaseModel):
    """System information where the submission was executed."""
    model_config = ConfigDict(from_attributes=True)
    
    cpu: Optional[str] = None
    gpu: Optional[str] = None
    platform: Optional[str] = None
    torch: Optional[str] = None
    device_count: Optional[int] = None
    runtime: Optional[str] = None


class Submission(BaseModel):
    """Pydantic model representing a single competition submission."""
    model_config = ConfigDict(from_attributes=True)
    
    submission_id: int
    leaderboard_id: int
    user_id: int
    submission_time: datetime.datetime
    file_name: str
    code: bytes
    code_id: int
    run_id: int
    run_start_time: Optional[datetime.datetime] = None
    run_end_time: Optional[datetime.datetime] = None
    run_mode: Optional[str] = None
    run_score: Optional[float] = None
    run_passed: Optional[bool] = None
    run_meta: Optional[RunMeta] = None
    run_system_info: Optional[RunSystemInfo] = None


class CompetitionDataset:
    """
    An extremely fast, memory-efficient API for large parquet files.
    Optimized for speed by using column projection and memory mapping.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        # memory_map=True allows the OS to handle file caching efficiently
        self.parquet_file = pq.ParquetFile(file_path, memory_map=True)
        self._total_rows = self.parquet_file.metadata.num_rows
        # Pre-fetch schema columns to avoid repeated metadata lookups
        self._columns = self.parquet_file.schema.names

    def __len__(self) -> int:
        """Returns the total number of entries instantly from metadata."""
        return self._total_rows

    def sample_submissions(self, batch_size: int = 10) -> List[Submission]:
        """
        Returns a sample of submissions extremely fast.
        
        Optimization: Uses column projection to ensure only the necessary columns 
        for the first few rows are read from the disk.
        """
        if self._total_rows == 0:
            return []
            
        # We explicitly pass the columns we want to read. 
        # Even though we want all of them, being explicit helps pyarrow's internal router.
        reader = self.parquet_file.iter_batches(
            batch_size=batch_size, 
            columns=self._columns
        )
        
        try:
            first_batch = next(reader)
            records = first_batch.to_pylist()
            return [Submission(**record) for record in records]
        except StopIteration:
            return []

    def __iter__(self) -> Iterator[Submission]:
        """
        Lazily iterates through the parquet file with a small batch size.
        """
        for batch in self.parquet_file.iter_batches(batch_size=10, columns=self._columns):
            for record in batch.to_pylist():
                yield Submission(**record)

    def iter_batches(self, batch_size: int = 10) -> Iterator[List[Submission]]:
        """
        Iterates through the dataset in batches.
        """
        for batch in self.parquet_file.iter_batches(batch_size=batch_size, columns=self._columns):
            yield [Submission(**record) for record in batch.to_pylist()]
