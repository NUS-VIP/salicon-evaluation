from distutils.core import setup

setup(
    author='Shane Huang',
    author_email = 'shannie.huang@gmail.com',
    name='pycocoevalsal',
    packages=['pycocoevalsal'],
    package_dir = {'pycocoevalsal': 'pycocoevalsal'},
    version='0.1',
    install_requires=['coco'],
    dependency_links=['https://github.com/pdollar/coco/tarball/master#egg=coco']    
    
)



