from __future__ import annotations

import datetime
from typing import List, Optional, Iterator, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
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
    
    id: int
    leaderboard_id: int
    user_id: str
    submission_time: datetime.datetime
    file_name: str
    code_id: int
    status: str
    done: Optional[bool] = None
    
    # Run information (joined from runs table)
    run_id: Optional[int] = None
    run_start_time: Optional[datetime.datetime] = None
    run_end_time: Optional[datetime.datetime] = None
    run_mode: Optional[str] = None
    run_score: Optional[float] = None
    run_passed: Optional[bool] = None
    run_meta: Optional[RunMeta] = None
    run_system_info: Optional[RunSystemInfo] = None


class CompetitionDataset:
    """
    An API for interacting with the competition database.
    Refactored to consume data from PostgreSQL instead of Parquet.
    """

    def __init__(self, database_url: str, schema: str = "leaderboard"):
        self.database_url = database_url
        self._conn = None
        self._schema = schema

    def _get_connection(self):
        """Internal method to manage and return a database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url, sslmode="require")
        return self._conn

    def __len__(self) -> int:
        """Returns the total number of submissions instantly from the database."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) FROM "{self._schema}"."submission";')
            return cur.fetchone()[0]

    def _get_submission_query(self) -> str:
        """Returns the base SQL query for submissions with joined run data."""
        return f"""
            SELECT 
                s.id, s.leaderboard_id, s.user_id, s.submission_time, s.file_name, s.code_id, s.status, s.done,
                r.id as run_id, r.start_time as run_start_time, r.end_time as run_end_time, 
                r.mode as run_mode, r.score as run_score, r.passed as run_passed,
                r.meta as run_meta, r.system_info as run_system_info
            FROM "{self._schema}"."submission" s
            LEFT JOIN "{self._schema}"."runs" r ON s.id = r.submission_id
        """

    def sample_submissions(self, batch_size: int = 10) -> List[Submission]:
        """
        Returns a sample of submissions.
        
        Optimization: Uses SQL LIMIT to ensure only the necessary records are fetched.
        """
        conn = self._get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = self._get_submission_query() + " LIMIT %s"
            cur.execute(query, (batch_size,))
            records = cur.fetchall()
            return [Submission(**dict(record)) for record in records]

    def __iter__(self) -> Iterator[Submission]:
        """
        Lazily iterates through all submissions using a server-side cursor.
        """
        conn = self._get_connection()
        # Named cursor ensures server-side fetching, avoiding loading everything into memory
        cursor_name = f"sub_iter_{id(self)}"
        with conn.cursor(name=cursor_name, cursor_factory=RealDictCursor) as cur:
            cur.execute(self._get_submission_query())
            while True:
                records = cur.fetchmany(size=10)
                if not records:
                    break
                for record in records:
                    yield Submission(**dict(record))

    def iter_batches(self, batch_size: int = 10) -> Iterator[List[Submission]]:
        """
        Iterates through the dataset in batches.
        """
        conn = self._get_connection()
        cursor_name = f"batch_iter_{id(self)}"
        with conn.cursor(name=cursor_name, cursor_factory=RealDictCursor) as cur:
            cur.execute(self._get_submission_query())
            while True:
                records = cur.fetchmany(size=batch_size)
                if not records:
                    break
                yield [Submission(**dict(record)) for record in records]

    def close(self):
        """Closes the database connection and cleans up resources."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None
