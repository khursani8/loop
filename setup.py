import setuptools


__packagename__ = 'utili'

def readme():
    with open('README.md') as f:
        return f.read()


with open('requirements.txt') as fopen:
    req = list(filter(None, fopen.read().split('\n')))

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='0.2',
    python_requires='>=3.8.*',
    description="Sani's helper for training deep learning model",
    long_description=readme(),
    author='khursani8',
    author_email='khursani8@gmail.com',
    keywords=['pytorch','utility','helper'],
    install_requires=req,
    license='MIT',
)