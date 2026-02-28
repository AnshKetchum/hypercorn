from __future__ import annotations
import os
import datetime
from typing import List, Optional, Iterator, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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


class Competition(BaseModel):
    """
    Pydantic model representing a single competition (row in leaderboard.leaderboard).
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    deadline: Optional[datetime.datetime] = None
    creator_id: Optional[int] = None
    forum_id: Optional[int] = None
    secret_seed: Optional[int] = None
    description: Optional[str] = None
    task: Optional[Dict[str, Any]] = None


class Submission(BaseModel):
    """
    Pydantic model representing a single competition submission.

    When the submission table is populated this model is backed by a JOIN of
    leaderboard.submission and leaderboard.runs.  The competition_name and
    competition_deadline fields are additionally resolved from the
    leaderboard.leaderboard table so that callers always have full context.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    leaderboard_id: int
    user_id: str
    submission_time: datetime.datetime
    file_name: str
    code_id: int
    status: str
    done: Optional[bool] = None

    # Enriched from leaderboard.leaderboard
    competition_name: Optional[str] = None
    competition_deadline: Optional[datetime.datetime] = None
    competition_description: Optional[str] = None

    # Run information (joined from leaderboard.runs)
    run_id: Optional[int] = None
    run_start_time: Optional[datetime.datetime] = None
    run_end_time: Optional[datetime.datetime] = None
    run_mode: Optional[str] = None
    run_score: Optional[float] = None
    run_passed: Optional[bool] = None
    run_meta: Optional[RunMeta] = None
    run_system_info: Optional[RunSystemInfo] = None


# ---------------------------------------------------------------------------
# CompetitionDataset
# ---------------------------------------------------------------------------

class CompetitionDataset:
    """
    API for interacting with the Kernelbot competition database.

    Data is read directly from PostgreSQL (schema: ``leaderboard`` by default).
    The primary tables are:

    * ``leaderboard.leaderboard``  – one row per competition / problem
    * ``leaderboard.submission``   – one row per user submission
    * ``leaderboard.runs``         – execution results for each submission
    * ``leaderboard.ranking_snapshot`` – periodic leaderboard snapshots
    * ``leaderboard.user_info``    – registered users

    Usage::

        db = CompetitionDataset(DATABASE_URL)

        # Competitions (backed by leaderboard.leaderboard – 5 sample rows)
        for comp in db.iter_competitions():
            print(comp)

        # Submissions (backed by leaderboard.submission + runs join)
        for sub in db.iter_submissions():
            print(sub)

        db.close()
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        schema: str = "leaderboard",
    ):
        self.database_url = database_url or os.getenv("KERNELBOT_API_URL", None)
        assert self.database_url
        self._schema = schema
        self._conn: Optional[psycopg2.extensions.connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Return an open database connection, reconnecting if necessary."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url, sslmode="require")
        return self._conn

    def close(self) -> None:
        """Close the database connection and release resources."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Internal SQL helpers
    # ------------------------------------------------------------------

    def _submission_query(self) -> str:
        """
        Base SQL that joins submission → runs → leaderboard to produce
        fully-enriched Submission rows.
        """
        s = self._schema
        return f"""
            SELECT
                sub.id,
                sub.leaderboard_id,
                sub.user_id,
                sub.submission_time,
                sub.file_name,
                sub.code_id,
                sub.status,
                sub.done,
                lb.name        AS competition_name,
                lb.deadline    AS competition_deadline,
                lb.description AS competition_description,
                r.id           AS run_id,
                r.start_time   AS run_start_time,
                r.end_time     AS run_end_time,
                r.mode         AS run_mode,
                r.score        AS run_score,
                r.passed       AS run_passed,
                r.meta         AS run_meta,
                r.system_info  AS run_system_info
            FROM "{s}"."submission" sub
            LEFT JOIN "{s}"."leaderboard" lb ON sub.leaderboard_id = lb.id
            LEFT JOIN "{s}"."runs"        r  ON sub.id = r.submission_id
        """

    @staticmethod
    def _build_submission(record: dict) -> Submission:
        """Convert a raw DB row dict into a Submission model instance."""
        run_meta = None
        if record.get("run_meta"):
            run_meta = RunMeta(**record["run_meta"])

        run_system_info = None
        if record.get("run_system_info"):
            run_system_info = RunSystemInfo(**record["run_system_info"])

        return Submission(
            id=record["id"],
            leaderboard_id=record["leaderboard_id"],
            user_id=record["user_id"],
            submission_time=record["submission_time"],
            file_name=record["file_name"],
            code_id=record["code_id"],
            status=record["status"],
            done=record.get("done"),
            competition_name=record.get("competition_name"),
            competition_deadline=record.get("competition_deadline"),
            competition_description=record.get("competition_description"),
            run_id=record.get("run_id"),
            run_start_time=record.get("run_start_time"),
            run_end_time=record.get("run_end_time"),
            run_mode=record.get("run_mode"),
            run_score=record.get("run_score"),
            run_passed=record.get("run_passed"),
            run_meta=run_meta,
            run_system_info=run_system_info,
        )

    # ------------------------------------------------------------------
    # Competition API  (leaderboard.leaderboard)
    # ------------------------------------------------------------------

    def competition_count(self) -> int:
        """Return the total number of competitions in the database."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) FROM "{self._schema}"."leaderboard";')
            return cur.fetchone()[0]

    def get_competitions(self, limit: Optional[int] = None) -> List[Competition]:
        """
        Return a list of Competition objects from ``leaderboard.leaderboard``.

        Parameters
        ----------
        limit:
            Maximum number of competitions to return.  ``None`` returns all.
        """
        conn = self._get_connection()
        sql = (
            f'SELECT id, name, deadline, creator_id, forum_id, '
            f'secret_seed, description, task '
            f'FROM "{self._schema}"."leaderboard" ORDER BY id'
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT %s"
            params = (limit,)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [Competition(**dict(row)) for row in cur.fetchall()]

    def iter_competitions(self, batch_size: int = 10) -> Iterator[Competition]:
        """
        Lazily iterate over all competitions using a server-side cursor.
        """
        conn = self._get_connection()
        cursor_name = f"comp_iter_{id(self)}"
        with conn.cursor(name=cursor_name, cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f'SELECT id, name, deadline, creator_id, forum_id, '
                f'secret_seed, description, task '
                f'FROM "{self._schema}"."leaderboard" ORDER BY id'
            )
            while True:
                rows = cur.fetchmany(size=batch_size)
                if not rows:
                    break
                for row in rows:
                    yield Competition(**dict(row))

    def get_competition_by_name(self, name: str) -> Optional[Competition]:
        """Return the Competition with the given name, or ``None`` if not found."""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f'SELECT id, name, deadline, creator_id, forum_id, '
                f'secret_seed, description, task '
                f'FROM "{self._schema}"."leaderboard" WHERE name = %s',
                (name,),
            )
            row = cur.fetchone()
            return Competition(**dict(row)) if row else None

    def get_competition_by_id(self, competition_id: int) -> Optional[Competition]:
        """Return the Competition with the given id, or ``None`` if not found."""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f'SELECT id, name, deadline, creator_id, forum_id, '
                f'secret_seed, description, task '
                f'FROM "{self._schema}"."leaderboard" WHERE id = %s',
                (competition_id,),
            )
            row = cur.fetchone()
            return Competition(**dict(row)) if row else None

    # ------------------------------------------------------------------
    # Submission API  (leaderboard.submission + runs join)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of submissions in the database."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) FROM "{self._schema}"."submission";')
            return cur.fetchone()[0]

    def sample_submissions(self, batch_size: int = 10) -> List[Submission]:
        """
        Return up to *batch_size* submissions, enriched with competition and
        run metadata.

        Uses SQL ``LIMIT`` so only the requested rows are transferred.
        """
        conn = self._get_connection()
        sql = self._submission_query() + " LIMIT %s"
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (batch_size,))
            return [self._build_submission(dict(r)) for r in cur.fetchall()]

    def get_submissions_for_competition(
        self, competition_id: int, limit: Optional[int] = None
    ) -> List[Submission]:
        """
        Return all submissions for a given competition, optionally capped at
        *limit* rows.
        """
        conn = self._get_connection()
        sql = self._submission_query() + " WHERE sub.leaderboard_id = %s ORDER BY sub.submission_time DESC"
        params: tuple = (competition_id,)
        if limit is not None:
            sql += " LIMIT %s"
            params = (competition_id, limit)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [self._build_submission(dict(r)) for r in cur.fetchall()]

    def __iter__(self) -> Iterator[Submission]:
        """
        Lazily iterate over all submissions using a server-side cursor.
        Memory-efficient: rows are streamed in small batches from the server.
        """
        conn = self._get_connection()
        cursor_name = f"sub_iter_{id(self)}"
        with conn.cursor(name=cursor_name, cursor_factory=RealDictCursor) as cur:
            cur.execute(self._submission_query())
            while True:
                rows = cur.fetchmany(size=10)
                if not rows:
                    break
                for row in rows:
                    yield self._build_submission(dict(row))

    def iter_submissions(self, batch_size: int = 10) -> Iterator[Submission]:
        """Alias for ``__iter__`` that makes intent explicit."""
        return iter(self)

    def iter_batches(self, batch_size: int = 10) -> Iterator[List[Submission]]:
        """
        Iterate over submissions in batches of *batch_size*.
        Each yielded value is a list of Submission objects.
        """
        conn = self._get_connection()
        cursor_name = f"batch_iter_{id(self)}"
        with conn.cursor(name=cursor_name, cursor_factory=RealDictCursor) as cur:
            cur.execute(self._submission_query())
            while True:
                rows = cur.fetchmany(size=batch_size)
                if not rows:
                    break
                yield [self._build_submission(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # User API  (leaderboard.user_info)
    # ------------------------------------------------------------------

    def get_users(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return user records from ``leaderboard.user_info``.

        Parameters
        ----------
        limit:
            Maximum number of users to return.  ``None`` returns all.
        """
        conn = self._get_connection()
        sql = f'SELECT id, user_name, cli_valid, created_at FROM "{self._schema}"."user_info" ORDER BY created_at'
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT %s"
            params = (limit,)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


