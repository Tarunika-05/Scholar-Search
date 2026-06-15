import requests

def test_rate_limit():
    url = "http://localhost:8000/query"
    headers = {"X-API-Key": "test-user-key"}
    payload = {"query": "deep learning"}

    print("Sending 65 requests to /query...")
    success_count = 0
    rate_limited_count = 0

    for i in range(65):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=2)
            if resp.status_code == 200:
                success_count += 1
            elif resp.status_code == 429:
                rate_limited_count += 1
            else:
                print(f"Unexpected status: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Request {i} failed: {e}")

    print(f"Successful requests (Expected ~60): {success_count}")
    print(f"Rate limited requests (Expected ~5): {rate_limited_count}")

if __name__ == "__main__":
    test_rate_limit()
