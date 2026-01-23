"""
Simple load testing script for the API.

Usage:
    python -m tests.load_test_api --url http://localhost:8000 --users 10 --requests 100

Can also be run with pytest:
    pytest tests/load_test_api.py -v
"""

import asyncio
import time
import argparse
from typing import List
import httpx
from PIL import Image
import io
import pytest


async def make_request(client: httpx.AsyncClient, url: str, image_bytes: io.BytesIO) -> dict:
    """Make a single prediction request."""
    image_bytes.seek(0)
    files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
    data = {"top_k": 5}

    start_time = time.time()
    try:
        response = await client.post(f"{url}/predict/upload", files=files, data=data, timeout=30.0)
        elapsed = time.time() - start_time

        return {
            "status_code": response.status_code,
            "elapsed": elapsed,
            "success": response.status_code == 200
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status_code": None,
            "elapsed": elapsed,
            "success": False,
            "error": str(e)
        }


async def run_load_test(url: str, num_users: int, requests_per_user: int):
    """Run load test with multiple concurrent users."""
    # Create a test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    image_bytes = img_bytes.getvalue()

    results: List[dict] = []
    start_time = time.time()

    async def user_task(user_id: int):
        """Task for a single user making requests."""
        async with httpx.AsyncClient() as client:
            user_results = []
            for i in range(requests_per_user):
                result = await make_request(
                    client,
                    url,
                    io.BytesIO(image_bytes)
                )
                result["user_id"] = user_id
                result["request_id"] = i
                user_results.append(result)
            return user_results

    # Create tasks for all users
    tasks = [user_task(i) for i in range(num_users)]
    user_results_list = await asyncio.gather(*tasks)

    # Flatten results
    for user_results in user_results_list:
        results.extend(user_results)

    total_time = time.time() - start_time

    # Calculate statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    total_requests = len(results)
    success_count = len(successful)
    failure_count = len(failed)

    if successful:
        avg_latency = sum(r["elapsed"] for r in successful) / len(successful)
        min_latency = min(r["elapsed"] for r in successful)
        max_latency = max(r["elapsed"] for r in successful)
    else:
        avg_latency = min_latency = max_latency = 0

    requests_per_second = total_requests / total_time if total_time > 0 else 0

    # Print results
    print("\n" + "="*60)
    print("LOAD TEST RESULTS")
    print("="*60)
    print(f"Total requests: {total_requests}")
    print(f"Successful: {success_count} ({100*success_count/total_requests:.1f}%)")
    print(f"Failed: {failure_count} ({100*failure_count/total_requests:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests per second: {requests_per_second:.2f}")
    print(f"\nLatency (successful requests):")
    print(f"  Average: {avg_latency*1000:.2f}ms")
    print(f"  Min: {min_latency*1000:.2f}ms")
    print(f"  Max: {max_latency*1000:.2f}ms")
    print("="*60)

    return {
        "total_requests": total_requests,
        "success_count": success_count,
        "failure_count": failure_count,
        "requests_per_second": requests_per_second,
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency
    }


def main():
    """Main entry point for load testing."""
    parser = argparse.ArgumentParser(description="Load test the API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=10, help="Requests per user")

    args = parser.parse_args()

    print(f"Starting load test:")
    print(f"  URL: {args.url}")
    print(f"  Concurrent users: {args.users}")
    print(f"  Requests per user: {args.requests}")
    print(f"  Total requests: {args.users * args.requests}")

    asyncio.run(run_load_test(args.url, args.users, args.requests))


if __name__ == "__main__":
    main()


# Pytest test functions
@pytest.mark.asyncio
async def test_load_test_single_request():
    """Test that a single request can be made successfully."""
    # This test requires the API to be running
    # Skip if API is not available
    url = "http://localhost:8000"

    try:
        async with httpx.AsyncClient() as client:
            # Try to connect to health endpoint first
            health_response = await client.get(f"{url}/health", timeout=5.0)
            if health_response.status_code != 200:
                pytest.skip("API is not available")
    except Exception:
        pytest.skip("API is not available")

    # Create a test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    image_bytes = img_bytes.getvalue()

    async with httpx.AsyncClient() as client:
        result = await make_request(
            client,
            url,
            io.BytesIO(image_bytes)
        )

    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["elapsed"] > 0


@pytest.mark.asyncio
async def test_load_test_multiple_users():
    """Test load test with multiple concurrent users (lightweight test)."""
    url = "http://localhost:8000"

    try:
        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"{url}/health", timeout=5.0)
            if health_response.status_code != 200:
                pytest.skip("API is not available")
    except Exception:
        pytest.skip("API is not available")

    # Run a lightweight load test: 2 users, 2 requests each
    results = await run_load_test(url, num_users=2, requests_per_user=2)

    assert results["total_requests"] == 4
    assert results["success_count"] > 0
    assert results["requests_per_second"] > 0
    assert results["avg_latency"] > 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_load_test_heavy_load():
    """Heavy load test - marked as slow, can be skipped with -m 'not slow'."""
    url = "http://localhost:8000"

    try:
        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"{url}/health", timeout=5.0)
            if health_response.status_code != 200:
                pytest.skip("API is not available")
    except Exception:
        pytest.skip("API is not available")

    # Run a heavier load test: 10 users, 10 requests each
    results = await run_load_test(url, num_users=10, requests_per_user=10)

    assert results["total_requests"] == 100
    # At least 90% success rate
    success_rate = results["success_count"] / results["total_requests"]
    assert success_rate >= 0.9, f"Success rate {success_rate:.2%} is below 90%"
