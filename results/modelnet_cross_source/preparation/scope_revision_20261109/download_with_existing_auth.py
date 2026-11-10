"""Transport wrapper: reuse the user's explicit credential without logging it.

The scientific loader, source revision, SHA checks and access boundary remain
in the sealed experiment. This wrapper does not apply for gated access.
"""
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from experiments import modelnet_cross_source as experiment


def existing_user_credential():
    from huggingface_hub import get_token
    token = get_token()
    if token:
        return token, 'existing_huggingface_environment_or_store'
    record = json.loads(Path(__file__).with_name('credential_source.json').read_text())
    if not record['existing_user_provided_credential_message_found']:
        return None, 'unavailable'
    with Path(record['session_path']).open() as stream:
        for line in stream:
            if 'hf_' not in line or '我提供了登陆凭据' not in line:
                continue
            value = json.loads(line)
            payload = value.get('payload', {})
            messages = []
            if value.get('type') == 'event_msg' and payload.get('type') == 'user_message':
                messages = [payload.get('message', '')]
            elif value.get('type') == 'response_item' and payload.get('role') == 'user':
                messages = [c.get('text', '') for c in payload.get('content', []) if isinstance(c, dict)]
            for message in messages:
                match = re.match(r'^我提供了登陆凭据[：:]\s*(hf_[A-Za-z0-9]+)', message.strip())
                if match:
                    return match.group(1), 'previous_explicit_user_message_in_this_thread'
    return None, 'unavailable'


if __name__ == '__main__':
    experiment.ready()
    token, origin = existing_user_credential()
    if token:
        os.environ['HF_TOKEN'] = token
    status_path = experiment.OUT / 'download_runtime.json'
    status = dict(pid=os.getpid(), started_unix=time.time(), credential_available=bool(token),
                  credential_origin=origin, credential_logged=False, proxy_used=False,
                  access_application_submitted=False, lock_sha256=experiment.sha(experiment.OUT/'lock.json'))
    experiment.write(status_path, dict(status, state='running'))
    print('DOWNLOAD_STARTED', 'credential_available', bool(token), flush=True)
    try:
        experiment.download()
    except Exception as error:
        # Do not print exception URLs, authorization headers, or signed redirects.
        experiment.write(status_path, dict(status, state='failed', finished_unix=time.time(),
                                           error_type=type(error).__name__))
        print('DOWNLOAD_FAILED', type(error).__name__, flush=True)
        raise SystemExit(1) from None
    finally:
        os.environ.pop('HF_TOKEN', None)
        token = None
    experiment.write(status_path, dict(status, state='complete', finished_unix=time.time()))
    print('DOWNLOAD_COMPLETE', flush=True)
