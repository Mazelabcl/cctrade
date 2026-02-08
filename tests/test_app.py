"""Basic tests to verify the app starts and core routes work."""


def test_health_endpoint(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'


def test_dashboard_loads(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Dashboard' in response.data


def test_data_status_loads(client):
    response = client.get('/data/')
    assert response.status_code == 200
    assert b'Data Status' in response.data


def test_charts_loads(client):
    response = client.get('/charts/')
    assert response.status_code == 200
    assert b'Chart' in response.data


def test_features_loads(client):
    response = client.get('/features/')
    assert response.status_code == 200


def test_models_loads(client):
    response = client.get('/models/')
    assert response.status_code == 200


def test_api_stats_empty(client):
    response = client.get('/api/stats')
    assert response.status_code == 200
    data = response.get_json()
    assert data['candle_count'] == 0
    assert data['level_count'] == 0


def test_api_candles_empty(client):
    response = client.get('/api/candles')
    assert response.status_code == 200
    data = response.get_json()
    assert data == []


def test_api_candles_with_data(client, sample_candles):
    response = client.get('/api/candles?tf=1h')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 10
    assert data[0]['symbol'] == 'BTCUSDT'


def test_api_levels_with_data(client, sample_levels):
    response = client.get('/api/levels')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 4


def test_dashboard_with_data(client, sample_candles, sample_levels):
    response = client.get('/')
    assert response.status_code == 200
    assert b'10' in response.data  # 10 candles
