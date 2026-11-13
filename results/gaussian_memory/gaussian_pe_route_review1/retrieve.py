"""Bounded HTTP evidence retrieval and HTML text extraction; never executes author code."""
import sys,json,os,time,hashlib,urllib.request,urllib.error,datetime
from pathlib import Path
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parent
class Text(HTMLParser):
 def __init__(self): super().__init__(); self.parts=[]; self.skip=0
 def handle_starttag(self,t,a):
  if t in ('script','style'): self.skip+=1
  if t in ('p','div','section','h1','h2','h3','h4','li','tr','br'): self.parts.append('\n')
 def handle_endtag(self,t):
  if t in ('script','style'): self.skip=max(0,self.skip-1)
 def handle_data(self,d):
  if not self.skip:self.parts.append(d)
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
url,name,kind=sys.argv[1:4]
log=ROOT/'retrieval_log.jsonl'; records=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
start=ROOT/'retrieval_start.json'
if not start.exists():start.write_text(json.dumps({'first_evidence_utc':now(),'initial_free_bytes':os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize}))
since=datetime.datetime.fromisoformat(json.loads(start.read_text())['first_evidence_utc'])
assert (datetime.datetime.now(datetime.timezone.utc)-since).total_seconds()<3600,'time cap'
assert len(records)<48,'request cap'
assert sum(r.get('kind')=='author_source' and r.get('ok',False) for r in records)<24 or kind!='author_source','source cap'
used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()); free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
assert used+8*1024**2<128*1024**2 and free-8*1024**2>1024**3,'space guard'
r={'id':len(records)+1,'url':url,'name':name,'kind':kind,'started_utc':now(),'transport':'urllib_https_no_automatic_redirect','ok':False}
try:
 req=urllib.request.Request(url,headers={'User-Agent':'GRoPE-bounded-primary-source-review/1.0','Accept':'text/html,application/json,text/plain,*/*'})
 with urllib.request.build_opener(NoRedirect).open(req,timeout=35) as f:
  data=f.read(4*1024**2+1);r['status']=f.status;r['content_type']=f.headers.get('Content-Type','')
  assert len(data)<=4*1024**2,'individual snapshot cap'
  out=ROOT/'evidence'/name;out.write_bytes(data)
  r.update(ok=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),snapshot=str(out.relative_to(ROOT)))
  if 'html' in r['content_type']:
   p=Text();p.feed(data.decode('utf-8',errors='replace'));out.with_suffix(out.suffix+'.txt').write_text(''.join(p.parts))
except Exception as e:
 r['error']=str(e)
 if isinstance(e,urllib.error.HTTPError):r['status']=e.code;r['location']=e.headers.get('Location')
r['ended_utc']=now();r['free_bytes_after']=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
with log.open('a') as f:f.write(json.dumps(r)+'\n')
print(json.dumps(r))
