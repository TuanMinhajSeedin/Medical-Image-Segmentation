import setuptools

#This helps if we doing pypi package. This looks for __init__ constructor file and install as local package
with open("README.md","r",encoding="utf-8") as f:
    long_description=f.read()

__version__="0.0.0"

REPO_NAME='Medical-Image-Segmentation'
AUTHOR_USER_NAME='TuanMinhajSeedin'
SRC_REPO='LiverTumorSegmentation'
AUTHOR_EMAIL='tuanminhajseedin@gmail.com'

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="python package for Liver Tumor Segmentation app",
    long_description=long_description,
    long_description_cotent="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker":f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src")

)