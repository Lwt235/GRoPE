"""Direct-only HF-Mirror transport for the unchanged, hash-pinned R65 download.

The base downloader still owns resume/hash checks and the full-population marker.
This adapter does not resume research, decode arrays, or change scientific sources.
"""
import argparse
from email.utils import parsedate_to_datetime
import fcntl
import math
import os
import random
import re
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from experiments import round65_acquire as base
from experiments.round65_common import OUT, PREP, checked, read, sha, write

ENDPOINT = 'https://hf-mirror.com'


def server_wait(headers, now):
    """Honor both HTTP Retry-After forms and HF's zero-quota reset timer."""
    delays = []
    retry = headers.get('Retry-After', '').strip()
    if re.fullmatch(r'\d+', retry):
        value = float(retry)
        if math.isfinite(value):
            delays.append((value, 'Retry-After'))
    elif retry:
        try:
            date = headers.get('Date')
            origin = parsedate_to_datetime(date).timestamp() if date else now
            delays.append((max(0., parsedate_to_datetime(retry).timestamp()-origin), 'Retry-After date'))
        except (ValueError, TypeError, OverflowError):
            pass
    for bucket in headers.get('RateLimit', '').split(','):
        remaining = re.search(r'(?:^|;)\s*r\s*=\s*(\d+)\s*(?:;|$)', bucket)
        reset = re.search(r'(?:^|;)\s*t\s*=\s*(\d+)\s*(?:;|$)', bucket)
        if remaining and reset and int(remaining[1]) == 0:
            value = float(reset[1])
            if math.isfinite(value):
                delays.append((value, 'RateLimit reset'))
    return max(delays, default=(None, 'exponential fallback'), key=lambda pair: pair[0])


class RequestGate:
    """One shared cooldown and paced request starts, including CDN redirects."""
    def __init__(self, gap=.25, clock=time.monotonic, sleep=time.sleep, jitter=None):
        self.gap, self.clock, self.sleep = gap, clock, sleep
        self.jitter = jitter or (lambda: random.uniform(.5, 2.))
        self.lock = threading.Lock()
        self.until = self.next_start = 0.
        self.last_limited = None
        self.streak = 0

    def before_request(self):
        while True:
            with self.lock:
                now = self.clock()
                delay = max(self.until, self.next_start)-now
                if delay <= 0:
                    self.next_start = now+self.gap
                    return
            # Release the lock while waiting so another worker can extend it.
            self.sleep(min(delay, 30.))

    def defer(self, headers, wall_now):
        requested, reason = server_wait(headers, wall_now)
        with self.lock:
            now = self.clock()
            if self.last_limited is None or now-self.last_limited > 600:
                self.streak = 0
            self.streak += 1
            self.last_limited = now
            delay = requested if requested is not None else min(300., 60.*2**min(self.streak-1, 3))
            delay = max(1., delay)+self.jitter()
            self.until = max(self.until, now+delay)
            return dict(wait_seconds=self.until-now, until_unix=wall_now+self.until-now,
                        reason=reason, all_workers=True)


class DirectMirror:
    def __init__(self, allowed_paths, directory):
        self.allowed_paths = set(allowed_paths)
        self.directory = directory
        self.lock = threading.Lock()
        self.gate = RequestGate()
        self.state = dict(proxy_policy='disabled for initial requests and all redirects',
                          network_requests=0, final_hosts={}, redirect_host_pairs={}, failures=0,
                          request_start_gap_seconds=self.gate.gap, rate_limit_events=0)
        owner = self

        class Redirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                target = urllib.parse.urlsplit(newurl)
                if target.scheme != 'https' or not target.hostname or target.username or target.password:
                    raise ValueError('unexpected non-HTTPS or credential-bearing redirect')
                pair = urllib.parse.urlsplit(req.full_url).hostname+' -> '+target.hostname
                with owner.lock:
                    owner.state['redirect_host_pairs'][pair] = owner.state['redirect_host_pairs'].get(pair, 0)+1
                owner.gate.before_request()
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        # An explicitly empty ProxyHandler also covers foreign CDN redirects.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirect())

    def request(self, original):
        prefix = f'https://huggingface.co/datasets/{base.REPO}/resolve/{base.REVISION}/'
        if not isinstance(original, urllib.request.Request) or not original.full_url.startswith(prefix):
            raise ValueError('transport accepts only the pinned dataset file requests')
        path = original.full_url[len(prefix):]
        if path not in self.allowed_paths or original.get_method() != 'GET' or original.data is not None:
            raise ValueError('request outside the fixed training allowlist')
        headers = dict(original.header_items())
        if any(k.lower() not in ('range',) for k in headers):
            raise ValueError('unexpected headers; this public-data download sends no credentials')
        return urllib.request.Request(ENDPOINT+original.full_url[len('https://huggingface.co'):], headers=headers)

    def urlopen(self, original, timeout=60):
        request = self.request(original)
        self.gate.before_request()
        with self.lock:
            self.state['network_requests'] += 1
        try:
            response = self.opener.open(request, timeout=timeout)
            host = urllib.parse.urlsplit(response.geturl()).hostname
            with self.lock:
                self.state['final_hosts'][host] = self.state['final_hosts'].get(host, 0)+1
                self.state['last_status'] = response.status
                self.state['last_content_range'] = response.headers.get('Content-Range')
                self.state['last_rate_headers'] = self.rate_headers(response.headers)
                self.state['observed_unix'] = time.time()
                write(self.directory/'transport.json', self.state)
            return response
        except Exception as error:
            cooldown = None
            error_host = None
            if isinstance(error, urllib.error.HTTPError):
                error_host = urllib.parse.urlsplit(error.geturl()).hostname
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                cooldown = self.gate.defer(error.headers, time.time())
            with self.lock:
                self.state['failures'] += 1
                self.state['last_error_type'] = type(error).__name__
                if isinstance(error, urllib.error.HTTPError):
                    # Never persist signed CDN URLs, cookies or authorization.
                    self.state['last_http_error'] = dict(status=error.code,
                        response_host=error_host,
                        headers=self.rate_headers(error.headers), unix=time.time())
                if cooldown:
                    self.state['rate_limit_events'] += 1
                    self.state['cooldown'] = cooldown
                    write(self.directory/('rate_limit_'+str(time.time_ns())+'.json'),
                          dict(self.state['last_http_error'], cooldown=cooldown))
                self.state['observed_unix'] = time.time()
                write(self.directory/'transport.json', self.state)
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            if cooldown:
                print('RATE_LIMIT_COOLDOWN', error_host,
                      round(cooldown['wait_seconds'], 1), 'seconds', cooldown['reason'], flush=True)
            raise

    @staticmethod
    def rate_headers(headers):
        return {key: headers[key] for key in ('Retry-After', 'RateLimit', 'RateLimit-Policy', 'Date')
                if headers.get(key) is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('probe', 'acquire'))
    parser.add_argument('--file-path')
    args = parser.parse_args()
    if args.action == 'probe' and not args.file_path:
        parser.error('probe requires a training file path')
    checked()
    source = read(PREP/'train_catalog.json')
    if source['repository'] != base.REPO or source['revision'] != base.REVISION:
        raise ValueError('pinned source identity')
    rows = {r['file_path']: r for r in source['rows']}
    if set(rows) != {r['file_path'] for r in base.training_rows()}:
        raise ValueError('training allowlist mismatch')
    if args.action == 'probe' and args.file_path not in rows:
        raise ValueError('probe outside training allowlist')
    # Process-local changes only: never source or rewrite the user's shell files.
    removed = []
    for key in list(os.environ):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            removed.append(key); os.environ.pop(key)
    os.environ.update(NO_PROXY='*', no_proxy='*', HF_ENDPOINT=ENDPOINT)
    with (PREP/'mirror_download.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = PREP/('mirror_transport_'+str(time.time_ns()))
        directory.mkdir()
        transport = DirectMirror(rows, directory)
        write(directory/'plan.json', dict(pid=os.getpid(), started_unix=time.time(), action=args.action,
            probe_file=args.file_path, endpoint=ENDPOINT, proxy_policy=transport.state['proxy_policy'],
            removed_proxy_environment_keys=removed, raw_root=str(base.RAW),
            adapter_sha256=sha(__file__), base_source_sha256=sha(base.__file__),
            catalog_sha256=sha(PREP/'train_catalog.json'), lock_sha256=sha(OUT/'lock.json'),
            research_remains_paused=True, vertices_decoded=False, test_access=False))
        # Redirect the base module's network calls only; its source and validators
        # remain unchanged. Opener redirects stay within the direct-only opener.
        base.urllib = types.SimpleNamespace(error=urllib.error, request=types.SimpleNamespace(
            Request=urllib.request.Request, urlopen=transport.urlopen))
        print('DIRECT_MIRROR_START', os.getpid(), str(directory), flush=True)
        if args.action == 'acquire':
            base.acquire()
        else:
            row = rows[args.file_path]; path = base.RAW/args.file_path
            initial_partial_bytes = path.with_suffix('.part').stat().st_size if path.with_suffix('.part').exists() else 0
            started = time.monotonic()
            result = base.download_with_retry(row, directory)
            elapsed = time.monotonic()-started
            verified = sha(path) == row['sha256']
            if not verified:
                raise ValueError('completed mirror probe byte identity mismatch')
            write(directory/'probe_result.json', dict(result=result, seconds=elapsed,
                source_sha256=row['sha256'], sha256_verified=verified,
                initial_partial_bytes=initial_partial_bytes, proxy_policy=transport.state['proxy_policy'],
                vertices_decoded=False, research_remains_paused=True))
            print('MIRROR_FILE_VERIFIED', args.file_path, row['size'], 'seconds', elapsed,
                  'resumed_bytes', initial_partial_bytes, flush=True)


if __name__ == '__main__':
    main()
