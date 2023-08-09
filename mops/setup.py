import setuptools

with open("readme.md", "r", encoding="utf-8") as file_name:
    long_description = file_name.read()

pkgs = {
    "required": [
        "numpy",
        "pandas",
        "scipy"
    ]
}

setuptools.setup(
    name="mops", 
    version="1.0",
    author="Awol Seid Ebrie",
    author_email="es.awol@gmail.com",
    description="Multi-Objective Power Scheduling Simulation Envionment for Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=pkgs["required"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    include_package_data=True
)
