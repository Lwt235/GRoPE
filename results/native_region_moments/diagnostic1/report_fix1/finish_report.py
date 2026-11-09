"""Resume report after a Matplotlib keyword compatibility error."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parents[1]
result=json.loads((OUT/'result.json').read_text())
checks=json.loads((OUT/'execution_checks.json').read_text())
fix=json.loads((OUT/'report_fix1/failure_and_fix.json').read_text())
assert checks['passed'] and hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest()==checks['result_sha256']
source=OUT/'check_and_report.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==fix['original_script_sha256']
errors=checks['errors']
medians={k:np.asarray([r['metrics'][k]['quantiles'][2] for r in result['records']]) for k in result['context_median_summary']}
section='import matplotlib\n'+source.read_text().split('import matplotlib\n',1)[1]
assert section.count('tick_labels=')==2
section=section.replace('tick_labels=','labels=')
exec(compile(section,str(source)+'#report-resume','exec'))
