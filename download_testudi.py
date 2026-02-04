import urllib.request
import requests

url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
try:
    urllib.request.urlretrieve(url, "test_dog.jpg")
    print("Downloaded test_dog.jpg")
except Exception as e:
    print(f"Failed with urllib: {e}")
    try:
        response = requests.get(url, timeout=30)
        with open("test_dog.jpg", "wb") as f:
            f.write(response.content)
        print("Downloaded test_dog.jpg with requests")
    except Exception as e2:
        print(f"Failed with requests: {e2}")
