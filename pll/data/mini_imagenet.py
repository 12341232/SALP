
import os, zipfile, tarfile, shutil, hashlib
from urllib.request import urlretrieve
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets import ImageFolder

MINI_IMAGENET_URLS = [
    # A couple of commonly mirrored mini-ImageNet archives (84x84). If one fails, try the next.
    # Users can also manually place extracted data at: <data_root>/miniimagenet/{train,val,test}/class_x/*.jpg
    "https://storage.googleapis.com/miniimagenet-public/mini-imagenet.tar.gz",
    "https://opendatalab.s3.amazonaws.com/miniimagenet/mini-imagenet.zip"
]

def _download_and_extract(root):
    os.makedirs(root, exist_ok=True)
    ok = False
    for url in MINI_IMAGENET_URLS:
        try:
            fname = os.path.join(root, os.path.basename(url))
            if not os.path.exists(fname):
                print(f"Downloading mini-ImageNet from {url} ...")
                urlretrieve(url, fname)
            print("Extracting:", fname)
            if fname.endswith(".zip"):
                with zipfile.ZipFile(fname, "r") as z:
                    z.extractall(root)
            elif fname.endswith(".tar.gz") or fname.endswith(".tgz"):
                with tarfile.open(fname, "r:gz") as t:
                    t.extractall(root)
            ok = True
            break
        except Exception as e:
            print("Failed to fetch from", url, "err:", e)
    if not ok:
        raise RuntimeError("Could not download mini-ImageNet automatically. "
                           "Please prepare it under <data_root>/miniimagenet/{train,val,test}/...")

def ensure_miniimagenet(root):
    base = os.path.join(root, "miniimagenet")
    # Check presence (train/val/test directories with class subfolders)
    expect = [os.path.join(base, s) for s in ["train","val","test"]]
    if not all(os.path.isdir(p) for p in expect):
        _download_and_extract(root)
    return base
