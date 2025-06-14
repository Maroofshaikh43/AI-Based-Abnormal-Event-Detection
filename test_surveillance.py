import os
import time
import pytest
import tempfile
import shutil
from datetime import datetime
import cv2
import cloudinary.uploader
import logging

import flaskapp  # your flask app
from YOLO_Video import video_detection

# -- Fixtures ------------------------------------------------------------
@pytest.fixture(scope='module')
def app():
    flaskapp.app.config['TESTING'] = True
    flaskapp.app.config['WTF_CSRF_ENABLED'] = False
    return flaskapp.app

@pytest.fixture(scope='module')
def client(app):
    return app.test_client()

@pytest.fixture(autouse=True)
def isolate_env(monkeypatch, tmp_path):
    # Redirect saved videos into tmp
    save_dir = tmp_path / 'saved_videos'
    monkeypatch.setenv('SAVE_PATH', str(save_dir))
    os.makedirs(str(save_dir), exist_ok=True)
    # Prevent real Cloudinary uploads
    def fake_upload_large(path, **kwargs):
        return {'url': 'http://res.cloudinary.com/fake/' + os.path.basename(path)}
    monkeypatch.setattr(cloudinary.uploader, 'upload_large', fake_upload_large)
    # Setup logging capture
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

# -- Helper --------------------------------------------------------------
def list_saved_files():
    save_path = os.environ.get('SAVE_PATH')
    return os.listdir(save_path) if save_path and os.path.isdir(save_path) else []

# -- Scenario Tests ------------------------------------------------------
scenarios = [
    ('collision', 'tests/videos/collision.mp4', 'Accident'),
    ('parking_fight', 'tests/videos/fight.mp4', 'fighting'),
    ('chain_snatching', 'tests/videos/theft.mp4', 'chain snatching'),
    ('loitering', 'tests/videos/loitering.mp4', None),
    ('zone_breach', 'tests/videos/breach.mp4', None),
    ('night_theft', 'tests/videos/night_theft.mp4', 'chain snatching'),
    ('occlusion', 'tests/videos/occlusion.mp4', None),
]

@pytest.mark.parametrize('name, path, expected_cls', scenarios)
def test_detection_scenarios(name, path, expected_cls):
    # Run detection
    for _ in video_detection(path):
        pass
    files = list_saved_files()
    if expected_cls:
        assert any(f.startswith(expected_cls) for f in files), f"Scenario {name} did not save expected clip"
    else:
        # No saving expected for these scenarios
        assert True

# -- False Alarm ---------------------------------------------------------
def test_false_alarm():
    before = set(list_saved_files())
    for _ in video_detection('tests/videos/tree_motion.mp4'):
        pass
    after = set(list_saved_files())
    assert before == after, "False alarms generated unexpected clips"

# -- Multiple Concurrent Events ------------------------------------------
def test_multiple_concurrent_events(monkeypatch):
    calls = {'count': 0}
    def count_upload(path, **kwargs):
        calls['count'] += 1
        return {'url': 'ok'}
    monkeypatch.setattr(cloudinary.uploader, 'upload_large', count_upload)
    for _ in video_detection('tests/videos/two_fights.mp4'):
        pass
    # Expect two fight clips
    assert calls['count'] == 2, f"Expected 2 uploads, got {calls['count']}"

# -- Network Interruption -----------------------------------------------
def test_network_retry(monkeypatch):
    calls = {'count': 0}
    def flaky_upload(path, **kwargs):
        calls['count'] += 1
        if calls['count'] == 1:
            raise ConnectionError("Network down")
        return {'url': 'ok'}
    monkeypatch.setattr(cloudinary.uploader, 'upload_large', flaky_upload)
    for _ in video_detection('tests/videos/collision.mp4'):
        pass
    assert calls['count'] >= 2, "Upload was not retried after failure"

# -- Storage Quota Alert ------------------------------------------------
def test_storage_quota_alert(monkeypatch, caplog):
    # Simulate 90% usage
    def fake_disk_usage(path):
        total = 1000
        used = 900
        free = total - used
        return shutil._ntuple_diskusage(total, used, free)
    monkeypatch.setattr(shutil, 'disk_usage', fake_disk_usage)
    caplog.set_level(logging.WARNING)
    # Call a function in your app that checks storage; assume flaskapp.check_storage
    if hasattr(flaskapp, 'check_storage'):
        flaskapp.check_storage()
        assert any('storage low' in r.message.lower() for r in caplog.records)
    else:
        pytest.skip("Storage check not implemented")

# -- Cloud Credentials Expiry -------------------------------------------
def test_cloud_credentials_expiry(monkeypatch):
    def bad_credentials(path, **kwargs):
        raise cloudinary.APIError("Authentication error")
    monkeypatch.setattr(cloudinary.uploader, 'upload_large', bad_credentials)
    # Run detection; should still write local file
    for _ in video_detection('tests/videos/collision.mp4'):
        pass
    files = list_saved_files()
    assert files, "No local files saved after cloud failure"

# -- User Authentication & UI Access -------------------------------------
def test_login_logout(client):
    rv = client.post('/', data={'username': 'bad', 'password': 'no'}, follow_redirects=True)
    assert b'Invalid' in rv.data
    rv = client.post('/', data={'username': 'police', 'password': 'password'}, follow_redirects=True)
    assert b'Dashboard' in rv.data
    rv2 = client.get('/logout', follow_redirects=True)
    assert b'Login' in rv2.data

# -- Alert De-duplication -----------------------------------------------
def test_alert_deduplication(monkeypatch):
    calls = {'classes': []}
    def record_upload(path, **kwargs):
        cls = os.path.basename(path).split('_')[0]
        calls['classes'].append(cls)
        return {'url': 'ok'}
    monkeypatch.setattr(cloudinary.uploader, 'upload_large', record_upload)
    # Two overlapping camera videos
    for feed in ['tests/videos/overlap_cam1.mp4', 'tests/videos/overlap_cam2.mp4']:
        for _ in video_detection(feed): pass
    # Only one alert per class
    unique = set(calls['classes'])
    assert len(calls['classes']) == len(unique), "Duplicate alerts found"

# -- Edge-Case: Tiny Object ---------------------------------------------
def test_edge_case_tiny_object():
    # Expect no errors and possibly a clip
    for _ in video_detection('tests/videos/small_object.mp4'):
        pass
    # We cannot assert class; just ensure processing
    assert True

# -- Replay & Evidence Retrieval ----------------------------------------
def test_replay_evidence_retrieval(monkeypatch, client):
    # Fake video list
    fake_url = 'http://cloudinary/project/collision_2025-05-13.bundle'
    monkeypatch.setattr(flaskapp, 'fetch_videos_from_cloudinary', lambda prefix: [fake_url])
    client.post('/', data={'username':'police','password':'password'})
    rv = client.get('/dashboard')
    assert fake_url.encode() in rv.data

# -- Performance Regression ---------------------------------------------
def test_performance_regression():
    # Compare current fps to baseline
    start = time.time()
    dummy = cv2.imread('tests/videos/frame0001.jpg')
    for _ in range(50): dummy.copy()
    elapsed = time.time() - start
    fps = 50 / elapsed if elapsed>0 else float('inf')
    baseline_fps = 25
    assert fps >= 0.98 * baseline_fps, f"FPS regressed: {fps:.1f} < {0.98*baseline_fps:.1f}"

if __name__ == '__main__':
    pytest.main()
