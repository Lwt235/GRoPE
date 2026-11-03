"""Read-only download monitor; ETA uses new verified bytes after monitor startup."""
import argparse
from collections import deque
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

PREP = Path(__file__).resolve().parents[1] / 'results/round65_actual_gs/preparation'
TZ = ZoneInfo('Asia/Shanghai')


def read(path):
    return json.loads(path.read_text())


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '等待有效速度样本'
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours}小时{minutes:02d}分{seconds:02d}秒'


class Rate:
    def __init__(self, window):
        self.window = window
        self.samples = deque()

    def update(self, now, size):
        if self.samples and size < self.samples[-1][1]:
            self.samples.clear()
        self.samples.append((now, size))
        while len(self.samples) > 2 and self.samples[1][0] <= now - self.window:
            self.samples.popleft()
        elapsed = now - self.samples[0][0]
        # Never divide the existing cache size by monitor uptime. Wait for
        # a useful observation period and include polls without completions.
        if elapsed < min(30, self.window) or elapsed <= 0:
            return None, elapsed
        return (size - self.samples[0][1]) / elapsed, elapsed


def latest_attempt():
    attempts = [p for p in PREP.glob('acquisition_*')
                if p.name.removeprefix('acquisition_').isdigit() and (p / 'plan.json').exists()]
    if not attempts:
        raise FileNotFoundError('未找到 acquisition_*/plan.json')
    return max(attempts, key=lambda p: int(p.name.split('_', 1)[1]))


def existing_complete_bytes(catalog, raw_root):
    """Metadata-only baseline; rehashing cached files is not new downloading."""
    root = Path(raw_root)
    total = 0
    for row in catalog['rows']:
        try:
            if (root / row['file_path']).stat().st_size == row['size']:
                total += row['size']
        except FileNotFoundError:
            pass
    return total


def transport_record(pid):
    for folder in sorted(PREP.glob('mirror_transport_*'), reverse=True):
        plan = folder / 'plan.json'
        if plan.exists() and read(plan).get('pid') == pid:
            return folder / 'transport.json'
    return None


def process_state(pid):
    """None means process visibility is unavailable, not that it has stopped."""
    if not Path('/proc').is_dir():
        return None
    try:
        args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        return (b'experiments.round65_mirror_acquire' in args or
                b'experiments.round65_acquire' in args) and b'acquire' in args
    except FileNotFoundError:
        return False
    except PermissionError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, help='指定 acquisition 目录；默认跟踪最新下载')
    parser.add_argument('--interval', type=float, default=5, help='刷新间隔，秒（默认5）')
    parser.add_argument('--window', type=float, default=300, help='速度统计窗口，秒（默认300）')
    parser.add_argument('--stale-after', type=float, default=120, help='多久没有完成文件提示停滞（默认120秒）')
    parser.add_argument('--once', action='store_true', help='只显示一次状态，单次快照不估算速度')
    parser.add_argument('--no-clear', action='store_true', help='保留每次刷新记录，适合重定向日志')
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.interval, args.window, args.stale_after)):
        parser.error('时间参数必须是有限正数')
    if args.window < args.interval:
        parser.error('--window 不能小于 --interval')
    catalog = read(PREP / 'train_catalog.json')
    total_files = len(catalog['rows'])
    previous_attempt, rate = None, None
    try:
        while True:
            attempt = args.attempt.resolve() if args.attempt else latest_attempt()
            plan = read(attempt / 'plan.json')
            if attempt != previous_attempt:
                rate = Rate(args.window)
                cached_bytes = existing_complete_bytes(catalog, plan['raw_root'])
                transport = transport_record(plan['pid'])
                previous_attempt = attempt
            if plan['catalog_sha256'] != (PREP / 'train_catalog.sha256').read_text().strip():
                raise ValueError('下载记录与当前数据目录清单不匹配')
            total_bytes = plan['expected_bytes']
            progress_path = attempt / 'progress.json'
            progress = read(progress_path) if progress_path.exists() else {}
            size = progress.get('verified_bytes', 0)
            count = progress.get('completed', 0)
            now = time.time()
            age = max(0, now - progress.get('unix', plan['started_unix']))
            reviewing_cache = size < cached_bytes
            if reviewing_cache:
                speed, observed = None, 0
            else:
                speed, observed = rate.update(time.monotonic(), size)
            complete = (attempt / 'completion.json').exists()
            failed = (attempt / 'failure.json').exists()
            alive = process_state(plan['pid'])
            cooldown = 0
            if transport and transport.exists():
                cooldown = max(0, read(transport).get('cooldown', {}).get('until_unix', 0)-now)
            if complete:
                status, count, size = '下载完成（全部文件已校验）', total_files, total_bytes
            elif failed:
                status = '下载失败：' + read(attempt / 'failure.json').get('error_type', '未知错误')
            elif alive is False:
                status = '下载进程已停止'
            elif cooldown > 0:
                status = f'限流冷却：暂缓新请求（剩余{math.ceil(cooldown)}秒）'
            elif age >= args.stale_after:
                status = '暂未新增已校验文件（可能仍在下载大文件或重试）'
            else:
                status = '下载中' if alive else '进程状态不可见；按进度文件监控'
                if reviewing_cache:
                    status += '（已有文件不计入速度，等待新增样本）'
            remaining = max(0, total_bytes - size)
            eta = remaining / speed if speed and speed > 0 else None
            # Do not show an optimistic ETA for failed, stopped or stale work.
            if failed or alive is False or age >= args.stale_after or cooldown > 0:
                eta = None
            if complete:
                eta = 0
            speed_text = f'{speed / 1e6:.2f} MB/s' if speed is not None else '采样中（约30秒）'
            if speed == 0:
                speed_text = '0 MB/s（统计窗口内无新增校验字节）'
            finish = ((datetime.fromtimestamp(now, TZ) + timedelta(seconds=eta)).strftime('%m-%d %H:%M:%S')
                      if eta is not None else '暂不估算')
            lines = [
                datetime.fromtimestamp(now, TZ).strftime('%Y-%m-%d %H:%M:%S 北京时间') + ' | ' + status,
                f'文件：{count:,} / {total_files:,} | 字节进度：{size / total_bytes:.2%}',
                f'已校验：{size / 1e9:.2f} / {total_bytes / 1e9:.2f} GB | 剩余：{remaining / 1e9:.2f} GB',
                f'近期校验吞吐：{speed_text}（观测约{observed:.0f}秒，窗口{args.window:g}秒）',
                f'预计剩余：{duration(eta)} | 预计完成：{finish} 北京时间',
                f'上次进度更新：{age:.0f}秒前 | PID：{plan["pid"]}',
                f'记录：{attempt}',
                '估计依据新增已校验字节；未计入在途部分文件。按 Ctrl+C 只退出监控。',
            ]
            if sys.stdout.isatty() and not args.no_clear:
                print('\033[2J\033[H', end='')
            print('\n'.join(lines), flush=True)
            if args.once or complete or failed or alive is False:
                return 1 if failed or (alive is False and not complete) else 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n已退出监控，下载进程继续独立运行。')
        return 0


if __name__ == '__main__':
    sys.exit(main())
