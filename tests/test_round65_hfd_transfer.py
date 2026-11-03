import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.round65_hfd_transfer import HFD, clean_environment, prepare_hfd, verify


class HfdPreparationTests(unittest.TestCase):
    def test_environment_does_not_use_proxy_or_mirror_credentials(self):
        with patch.dict(os.environ, {'HTTP_PROXY':'secret', 'all_proxy':'secret',
                                     'HF_TOKEN':'secret', 'HF_USERNAME':'secret'}):
            env=clean_environment('/tmp/example-bin')
        self.assertEqual(env['NO_PROXY'], '*')
        for key in ('HTTP_PROXY','all_proxy','HF_TOKEN','HF_USERNAME'):
            self.assertNotIn(key, env)

    def test_cached_hfd_manifest_reuses_complete_files_and_filters_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);destination=root/'files';bin_dir=root/'bin'
            rows=[dict(file_path='0/a.ply',size=32,sha256=hashlib.sha256(b'x'*32).hexdigest()),
                  dict(file_path='1/b.ply',size=16,sha256=hashlib.sha256(b'x'*16).hexdigest())]
            command,env=prepare_hfd(destination,rows,bin_dir)
            first=destination/'0/a.ply';first.parent.mkdir();first.write_bytes(b'x'*32)
            inode=first.stat().st_ino
            # Any metadata request would fail rather than access the network.
            curl=bin_dir/'curl';curl.write_text('#!/bin/sh\nexit 88\n');curl.chmod(0o700)
            aria=bin_dir/'aria2c'
            aria.write_text("#!/home/liwt/miniColmap/.venv/bin/python\n"
                "from pathlib import Path\n"
                "p=Path('.hfd/aria2c_urls.txt');s=p.read_text()\n"
                "assert '1/b.ply' in s and '0/a.ply' not in s\n"
                "assert 'https://hf-mirror.com/datasets/zxAi/GSModel60/resolve/' in s\n"
                "Path('1').mkdir(exist_ok=True);Path('1/b.ply').write_bytes(b'x'*16)\n")
            aria.chmod(0o700)
            result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(first.stat().st_ino,inode)
            verify(destination,rows)

    def test_preallocated_or_wrong_hash_files_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);p=root/'0/a.ply';p.parent.mkdir();p.write_bytes(b'x'*32)
            row=dict(file_path='0/a.ply',size=32,sha256=hashlib.sha256(b'x'*32).hexdigest())
            side=p.with_suffix('.ply.aria2');side.write_bytes(b'control')
            with self.assertRaises(ValueError):verify(root,[row])
            side.unlink();p.write_bytes(b'y'*32)
            with self.assertRaises(ValueError):verify(root,[row])

    def test_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                prepare_hfd(Path(temporary)/'files',[dict(file_path='../test.ply',size=1)],
                            Path(temporary)/'bin')


if __name__=='__main__':unittest.main()
