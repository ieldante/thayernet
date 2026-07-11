"""Offline tests for conservative DR10 downloader behavior."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from scripts import download_dr10_grz_cutouts as downloader
from scripts.download_dr10_grz_cutouts import (
    FailureCircuitBreaker,
    GlobalRateLimiter,
    JsonlLogger,
    atomic_rename_noreplace,
    build_request_url,
    download_one,
    output_filename,
    retry_after_seconds,
    sanitized_argv,
    validate_fits,
)


class DownloaderContractTests(unittest.TestCase):
    def test_manual_smoke_fits_passes_semantic_validation(self) -> None:
        path = Path(
            "data/dr10_grz_cutouts/manual_smoke/"
            "ra190.1086_dec1.2005_grz_256.fits"
        )
        result = validate_fits(
            path,
            256,
            "grz",
            expected_ra=190.1086,
            expected_dec=1.2005,
        )
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.shape, "3x256x256")
        self.assertEqual(result.bands_header, "grz")
        self.assertEqual(result.finite_fraction, 1.0)
        self.assertLessEqual(result.center_offset_pixels, 0.05)

    def test_wrong_requested_coordinate_is_rejected_by_wcs_center(self) -> None:
        path = Path(
            "data/dr10_grz_cutouts/manual_smoke/"
            "ra190.1086_dec1.2005_grz_256.fits"
        )

        result = validate_fits(
            path,
            256,
            "grz",
            expected_ra=190.1186,
            expected_dec=1.2005,
        )

        self.assertFalse(result.valid)
        self.assertIn("WCS center offset", result.error)

    def test_html_saved_with_fits_suffix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "error.fits"
            path.write_text("<html>service error</html>", encoding="utf-8")
            result = validate_fits(path, 256, "grz")
        self.assertFalse(result.valid)
        self.assertIn("SIMPLE", result.error)

    def test_filename_and_url_are_deterministic(self) -> None:
        row = {
            "source_id": "100_2",
            "catalog_row_index": "17",
            "ra": "190.1086",
            "dec": "1.2005",
        }
        self.assertEqual(
            output_filename(row),
            "row17_100_2_ra190.1086000_dec+1.2005000_grz.fits",
        )
        url = build_request_url(
            "https://example.invalid/fits",
            {"ra": "190.1086", "dec": "1.2005", "bands": "grz"},
        )
        self.assertEqual(
            url,
            "https://example.invalid/fits?ra=190.1086&dec=1.2005&bands=grz",
        )

    def test_retry_after_numeric_seconds(self) -> None:
        self.assertEqual(retry_after_seconds("12"), 12.0)
        self.assertEqual(retry_after_seconds("-3"), 0.0)
        self.assertIsNone(retry_after_seconds("not-a-date"))

    def test_atomic_finalization_never_replaces_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.part"
            second = root / "second.part"
            destination = root / "science.fits"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            atomic_rename_noreplace(first, destination)
            with self.assertRaises(FileExistsError):
                atomic_rename_noreplace(second, destination)
            self.assertEqual(destination.read_bytes(), b"first")
            self.assertEqual(second.read_bytes(), b"second")

    def test_missing_zero_band_is_rejected(self) -> None:
        smoke = Path(
            "data/dr10_grz_cutouts/manual_smoke/"
            "ra190.1086_dec1.2005_grz_256.fits"
        )
        with fits.open(smoke) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32).copy()
            header = hdul[0].header.copy()
        data[0] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_g.fits"
            fits.PrimaryHDU(data=data, header=header).writeto(path)
            result = validate_fits(path, 256, "grz")
        self.assertFalse(result.valid)
        self.assertIn("blank or missing band", result.error)

    def test_circuit_breaker_opens_only_after_sustained_failures(self) -> None:
        breaker = FailureCircuitBreaker(3)
        self.assertFalse(breaker.record_failure())
        self.assertFalse(breaker.record_failure())
        breaker.record_success()
        self.assertFalse(breaker.record_failure())
        self.assertFalse(breaker.record_failure())
        self.assertTrue(breaker.record_failure())
        self.assertTrue(breaker.is_open())

    def test_limiter_wait_is_cooperatively_cancelled(self) -> None:
        limiter = GlobalRateLimiter(10.0)
        self.assertTrue(limiter.wait())
        cancel_event = threading.Event()
        result: list[bool] = []
        waiter = threading.Thread(
            target=lambda: result.append(limiter.wait(cancel_event)), daemon=True
        )
        waiter.start()
        time.sleep(0.02)
        cancel_event.set()
        limiter.wake_all()
        waiter.join(timeout=1.0)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [False])

    def test_sanitized_argv_redacts_conventional_secret_values(self) -> None:
        result = sanitized_argv(
            ["script.py", "--token", "secret-a", "--api-key=secret-b", "--resume"]
        )

        self.assertEqual(
            result,
            ["script.py", "--token", "<redacted>", "--api-key=<redacted>", "--resume"],
        )


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        body: bytes = b"",
        content_type: str = "application/fits",
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [self.body]

    def raise_for_status(self) -> None:
        return None


class DownloaderControlFlowTests(unittest.TestCase):
    @staticmethod
    def _row() -> dict[str, str]:
        return {
            "source_id": "100_2",
            "catalog_row_index": "17",
            "ra": "190.1086",
            "dec": "1.2005",
        }

    def _download(
        self,
        directory: str,
        breaker: FailureCircuitBreaker,
        cancel_event: threading.Event,
    ) -> dict[str, object]:
        return download_one(
            self._row(),
            output_dir=Path(directory),
            endpoint="https://example.invalid/fits",
            layer="ls-dr10-south",
            bands="grz",
            pixscale=0.262,
            size=256,
            timeout=(0.1, 0.1),
            max_retries=5,
            backoff_base=0.001,
            backoff_cap=0.001,
            user_agent="offline-test",
            resume=True,
            limiter=GlobalRateLimiter(0.0),
            circuit_breaker=breaker,
            cancel_event=cancel_event,
            event_log=JsonlLogger(Path(directory) / "events.jsonl"),
            run_id="offline-test",
        )

    def test_ordinary_http_4xx_is_terminal_and_does_not_open_circuit(self) -> None:
        breaker = FailureCircuitBreaker(1)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            downloader.requests,
            "get",
            return_value=_FakeResponse(404, content_type="text/plain"),
        ) as request:
            result = self._download(directory, breaker, threading.Event())

        self.assertEqual(result["status"], "http_client_error")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(request.call_count, 1)
        self.assertFalse(breaker.is_open())

    def test_semantic_fits_rejection_is_not_retried_or_circuited(self) -> None:
        smoke = Path(
            "data/dr10_grz_cutouts/manual_smoke/"
            "ra190.1086_dec1.2005_grz_256.fits"
        )
        with fits.open(smoke) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32).copy()
            header = hdul[0].header.copy()
        data[0] = 0.0
        breaker = FailureCircuitBreaker(1)
        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "invalid_payload.fits"
            fits.PrimaryHDU(data=data, header=header).writeto(payload_path)
            response = _FakeResponse(200, body=payload_path.read_bytes())
            with patch.object(downloader.requests, "get", return_value=response) as request:
                result = self._download(directory, breaker, threading.Event())

        self.assertEqual(result["status"], "validation_rejected")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(request.call_count, 1)
        self.assertFalse(breaker.is_open())

    def test_pre_cancelled_download_makes_no_http_request(self) -> None:
        breaker = FailureCircuitBreaker(1)
        cancel_event = threading.Event()
        cancel_event.set()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            downloader.requests, "get"
        ) as request:
            result = self._download(directory, breaker, cancel_event)

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["attempts"], 0)
        request.assert_not_called()
        self.assertFalse(breaker.is_open())

    def test_cancellation_during_backoff_prevents_retry_and_circuit_failure(self) -> None:
        breaker = FailureCircuitBreaker(1)
        cancel_event = threading.Event()

        def fail_once(*_args: object, **_kwargs: object) -> object:
            cancel_event.set()
            raise downloader.requests.Timeout("offline timeout")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            downloader.requests, "get", side_effect=fail_once
        ) as request:
            result = self._download(directory, breaker, cancel_event)

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(request.call_count, 1)
        self.assertFalse(breaker.is_open())

    def test_single_worker_sink_failure_does_not_start_second_row(self) -> None:
        calls: list[str] = []

        def fake_download(row: dict[str, str], **_kwargs: object) -> dict[str, object]:
            calls.append(row["source_id"])
            return {"status": "downloaded_valid", "bytes": 1}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "sources.csv"
            with manifest_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["source_id", "catalog_row_index", "ra", "dec"],
                )
                writer.writeheader()
                for index in range(5):
                    writer.writerow(
                        {
                            "source_id": f"source-{index}",
                            "catalog_row_index": index,
                            "ra": 190.0 + index * 0.01,
                            "dec": 1.0,
                        }
                    )
            argv = [
                "download_dr10_grz_cutouts.py",
                str(manifest_path),
                "--output-dir",
                str(root / "downloads"),
                "--workers",
                "1",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(downloader, "download_one", side_effect=fake_download),
                patch.object(
                    downloader.AppendOnlyCsv,
                    "append",
                    side_effect=RuntimeError("durable sink failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "durable sink failed"),
            ):
                downloader.main()

        self.assertEqual(calls, ["source-0"])

    def test_dry_run_records_complete_controls_and_one_worker_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "sources.csv"
            with manifest_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ra", "dec"])
                writer.writeheader()
                writer.writerow({"ra": 190.0, "dec": 1.0})
            argv = [
                "download_dr10_grz_cutouts.py",
                str(manifest_path),
                "--output-dir",
                str(root / "unused"),
                "--dry-run",
                "--resume",
                "--limit",
                "1",
                "--backoff-base",
                "3",
                "--backoff-cap",
                "11",
                "--checkpoint-every",
                "7",
            ]
            output = io.StringIO()
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
                self.assertEqual(downloader.main(), 0)
            payload, _end = json.JSONDecoder().raw_decode(output.getvalue())

        configuration = payload["configuration"]
        self.assertEqual(configuration["workers"], 1)
        self.assertEqual(configuration["request_interval"], 1.0)
        self.assertTrue(configuration["resume"])
        self.assertEqual(configuration["limit"], 1)
        self.assertEqual(configuration["backoff_base"], 3.0)
        self.assertEqual(configuration["backoff_cap"], 11.0)
        self.assertEqual(configuration["checkpoint_every"], 7)
        self.assertEqual(payload["argv"], argv)


if __name__ == "__main__":
    unittest.main()
