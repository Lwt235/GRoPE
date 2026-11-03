"""Exercise resumed-byte integrity without any remote dataset access."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
from experiments import round65_acquire as acquire


class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data)
        self.status=status
        self.headers=headers or {}


class AcquisitionTests(unittest.TestCase):
    def exercise(self,mode):
        payload=b'complete synthetic native source bytes'
        row=dict(file_path='0/example.ply',size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'raw';(root/'0').mkdir(parents=True)
            prefix=payload[:9]
            partial=root/'0/example.part';partial.write_bytes(prefix)
            def request(req,timeout):
                self.assertEqual(req.headers.get('Range'),'bytes=9-')
                if mode=='resume':
                    return Response(payload[9:],206,{'Content-Range':f'bytes 9-{len(payload)-1}/{len(payload)}'})
                if mode=='ignored':
                    return Response(payload)
                return Response(b'x'*(len(payload)-9),206,{'Content-Range':f'bytes 9-{len(payload)-1}/{len(payload)}'})
            with patch.object(acquire,'RAW',root),patch.object(acquire,'RESERVE',0),patch.object(acquire.urllib.request,'urlopen',request):
                if mode=='bad':
                    with self.assertRaisesRegex(ValueError,'identity mismatch'):acquire.download_one(row)
                    self.assertFalse((root/'0/example.ply').exists());self.assertTrue(partial.exists())
                else:
                    result=acquire.download_one(row)
                    self.assertEqual(result['bytes'],len(payload))
                    self.assertEqual((root/'0/example.ply').read_bytes(),payload)
                    if mode=='ignored':
                        archived=list((root/'0').glob('*.interrupted_*'))
                        self.assertEqual(len(archived),1);self.assertEqual(archived[0].read_bytes(),prefix)

    def test_resume_verified(self):self.exercise('resume')
    def test_server_ignoring_range_preserves_partial(self):self.exercise('ignored')
    def test_bad_response_never_becomes_completed_file(self):self.exercise('bad')

    def test_network_retry_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(acquire,'download_one',side_effect=[urllib.error.URLError('synthetic timeout'),{'done':True}]),patch.object(acquire.time,'sleep'):
                self.assertEqual(acquire.download_with_retry({'file_path':'0/a.ply'},Path(folder)),{'done':True})
                self.assertTrue((Path(folder)/'network_0_a.ply_0.json').is_file())

    def test_short_http_response_resumes_remaining_prefix(self):
        payload=b'complete synthetic native source bytes'
        row=dict(file_path='0/example.ply',size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'raw';(root/'0').mkdir(parents=True)
            (root/'0/example.part').write_bytes(payload[:9]);ranges=[]
            def request(req,timeout):
                offset=int(req.headers['Range'].split('=')[1].split('-')[0]);ranges.append(offset)
                body=payload[offset:18] if len(ranges)==1 else payload[offset:]
                return Response(body,206,{'Content-Range':f'bytes {offset}-{len(payload)-1}/{len(payload)}'})
            with patch.object(acquire,'RAW',root),patch.object(acquire,'RESERVE',0),patch.object(acquire.urllib.request,'urlopen',request),patch.object(acquire.time,'sleep'):
                acquire.download_with_retry(row,Path(folder))
            self.assertEqual(ranges,[9,18]);self.assertEqual((root/'0/example.ply').read_bytes(),payload)
            self.assertTrue((Path(folder)/'network_0_example.ply_0.json').exists())


if __name__=='__main__':unittest.main()
