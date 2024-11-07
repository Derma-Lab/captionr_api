from setuptools import setup, find_packages

setup(
    name="captionr",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision", 
        "Pillow",
        "requests",
        "tqdm",
        "open_clip_torch",
        "numpy",
        "thefuzz",
        "python-levenshtein"
    ],
    package_data={
        'captionr': ['data/*']
    },
    entry_points={
        'console_scripts': [
            'captionr=captionr:main',
        ],
    },
    author="theovercomer8",
    description="A tool for generating captions for images using CLIP",
    long_description="A tool for generating captions for images using CLIP",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    build_system={
        "requires": ["uv"],
        "build-backend": "uv.builders.python"
    }
)
