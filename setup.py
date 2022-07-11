import setuptools


__packagename__ = 'utiliti'

def readme():
    with open('README.md') as f:
        return f.read()


with open('requirements.txt') as fopen:
    req = list(filter(None, fopen.read().split('\n')))

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='0.1',
    python_requires='>=3.8.*',
    description="Sani's helper for training deep learning model",
    long_description=readme(),
    author='khursani8',
    author_email='khursani8@gmail.com',
    # url='https://github.com/huseinzol05/Malaya',
    # download_url='https://github.com/huseinzol05/Malaya/archive/master.zip',
    keywords=['pytorch','utility','helper'],
    install_requires=req,
    # extras_require={
    #     'gpu': ['tensorflow-gpu>=1.15'],
    #     'cpu': ['tensorflow>=1.15'],
    #     '': ['tensorflow>=1.15']
    # },
    license='MIT',
    # classifiers=[
    #         'Programming Language :: Python :: 3.6',
    #         'Intended Audience :: Developers',
    #         'License :: OSI Approved :: MIT License',
    #         'Operating System :: OS Independent',
    #         'Topic :: Text Processing',
    # ],
    # package_data={
    #     'malaya': [
    #         'function/web/*.html',
    #         'function/web/static/*.js',
    #         'function/web/static/*.css',
    #     ]},
)