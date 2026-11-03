"""Offline checks for shared throttling, server timers and safe diagnostics."""
from email.message import Message
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock

from experiments.round65_mirror_acquire import DirectMirror, RequestGate, server_wait
from experiments import round65_acquire as base


class FakeClock:
    def __init__(self):
        self.value = 0.
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class MirrorThrottleTests(unittest.TestCase):
    def test_retry_after_seconds_and_rate_reset_use_longest(self):
        self.assertEqual(server_wait({'Retry-After': '12', 'RateLimit': '"resolvers";r=0;t=90'}, 0),
                         (90., 'RateLimit reset'))

    def test_retry_after_date_uses_server_date_for_clock_skew(self):
        headers = {'Retry-After': 'Tue, 03 Nov 2026 03:02:00 GMT',
                   'Date': 'Tue, 03 Nov 2026 03:00:00 GMT'}
        self.assertEqual(server_wait(headers, 0), (120., 'Retry-After date'))

    def test_only_exhausted_buckets_contribute(self):
        headers = {'RateLimit': '"api";r=5;t=400, "resolvers";r=0;t=42'}
        self.assertEqual(server_wait(headers, 0), (42., 'RateLimit reset'))

    def test_invalid_headers_fall_back(self):
        self.assertEqual(server_wait({'Retry-After': 'nonsense', 'RateLimit': 'bad'}, 0),
                         (None, 'exponential fallback'))

    def test_long_server_timer_is_not_shortened_by_fallback_cap(self):
        clock = FakeClock()
        gate = RequestGate(clock=clock.now, sleep=clock.sleep, jitter=lambda: 0)
        gate.defer({'Retry-After': '900'}, 1000)
        # Another worker lacking headers cannot shorten the shared deadline.
        gate.defer({}, 1000)
        gate.before_request()
        self.assertEqual(clock.value, 900)
        gate.before_request()
        self.assertEqual(clock.value, 900.25)
        self.assertLessEqual(max(clock.sleeps), 30)

    def test_unknown_quota_has_exponential_fallback(self):
        clock = FakeClock()
        gate = RequestGate(clock=clock.now, sleep=clock.sleep, jitter=lambda: 0)
        self.assertEqual(gate.defer({}, 100)['wait_seconds'], 60)
        self.assertEqual(gate.defer({}, 100)['wait_seconds'], 120)
        self.assertEqual(gate.defer({}, 100)['wait_seconds'], 240)
        self.assertEqual(gate.defer({}, 100)['wait_seconds'], 300)

    def test_cooldown_extension_is_seen_by_waiting_workers(self):
        clock = FakeClock()
        gate = RequestGate(clock=clock.now, sleep=clock.sleep, jitter=lambda: 0)
        gate.defer({'Retry-After': '40'}, 1000)
        def sleep(seconds):
            clock.sleep(seconds)
            if clock.value == 30:
                gate.defer({'Retry-After': '100'}, 1030)
        gate.sleep = sleep
        gate.before_request()
        self.assertEqual(clock.value, 130)

    def test_http_error_records_host_and_headers_then_gate_delays_next_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            transport = DirectMirror({'0/example.ply'}, directory)
            clock = FakeClock()
            transport.gate = RequestGate(clock=clock.now, sleep=clock.sleep, jitter=lambda: 0)
            headers = Message()
            headers['Retry-After'] = '70'
            headers['Set-Cookie'] = 'secret-cookie'
            error = urllib.error.HTTPError('https://cdn.example/file?signature=secret-value',
                                           429, 'Too Many Requests', headers, io.BytesIO(b''))
            transport.opener.open = Mock(side_effect=error)
            url = f'https://huggingface.co/datasets/{base.REPO}/resolve/{base.REVISION}/0/example.ply'
            request = urllib.request.Request(url, headers={'Range': 'bytes=123-'})
            with self.assertRaises(urllib.error.HTTPError):
                transport.urlopen(request)
            record = json.loads((directory/'transport.json').read_text())
            self.assertEqual(record['last_http_error']['response_host'], 'cdn.example')
            self.assertEqual(record['cooldown']['wait_seconds'], 70)
            persisted = ''.join(p.read_text() for p in directory.glob('*.json'))
            self.assertNotIn('secret', persisted)
            response = Mock(status=206)
            response.headers = {'Content-Range': 'bytes 123-199/200'}
            response.geturl.return_value = 'https://cdn.example/file?signature=another-secret'
            transport.opener.open.side_effect = None
            transport.opener.open.return_value = response
            self.assertIs(transport.urlopen(request), response)
            self.assertEqual(clock.value, 70)
            sent = transport.opener.open.call_args.args[0]
            self.assertEqual(sent.get_header('Range'), 'bytes=123-')
            self.assertTrue(sent.full_url.startswith('https://hf-mirror.com/'))

    def test_credentials_and_unlisted_files_remain_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            transport = DirectMirror({'0/example.ply'}, Path(temporary))
            prefix = f'https://huggingface.co/datasets/{base.REPO}/resolve/{base.REVISION}/'
            with self.assertRaises(ValueError):
                transport.request(urllib.request.Request(prefix+'test.json'))
            with self.assertRaises(ValueError):
                transport.request(urllib.request.Request(prefix+'0/example.ply', headers={'Authorization':'secret'}))


if __name__ == '__main__':
    unittest.main()
